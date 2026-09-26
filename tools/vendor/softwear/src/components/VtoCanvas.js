// src/components/VtoCanvas.js
import React, { useRef, useEffect, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { VtoPoseEngine } from '../vto/VtoPoseEngine';
import { HeadPoseMapper } from '../vto/HeadPoseMapper';
import { GarmentPhysics } from '../vto/garmentPhysics';
import { SMPLXPoseMapper } from '../vto/SMPLXBoneMapper';
import { useStateManager } from '../stateManager';
import { useDeviceDetection } from '../utils/DeviceDetectionContext';
import { resolveModelPathWithFallback } from '../utils/modelPath';
import smplxBoneData from '../../public/data/smplx_bone_data.json';
import { configureUvBake, loadRetailImages, applyRetailUv, clearRetailUv, disposeRetailImages } from '../../../../retail3d/uvBake';
configureUvBake(THREE);
if (!window.softWearPerformance) {
    window.softWearPerformance = { garmentFPS: 0 };
}
// The tracked MediaPipe shoulder landmarks are anatomical shoulder JOINTS,
// the same points as the rig's shoulder joints, so the depth-matching span
// must be the RIG JOINT DISTANCE. MediaPipe sits a touch inside the true
// joints on near-frontal poses; a small fudge keeps the cloth from
// pinching. Never size from cloth/sleeve extents: sleeve-tip span is arm
// span, several times the joint distance, and oversizes the torso panel.
const RIG_SPAN_FUDGE = 1.06;
// Vertical bias as a fraction of the garment's own height, applied
// upward from the measured cloth shoulder line: MediaPipe's tracked
// joint line sits a little low relative to where a jacket's seams read
// on the body. Calibrated on the pixel scorer (collar/band windows).
// Calibrated on the pixel scorer against the true chin (the round-2 value
// 0.13 was derived from a glow-corrupted neck base and hung the collar band
// ~25px low).
// Calibrated on the pixel scorer against the true chin: the collar band
// must land at chin bottom + a few px (0.04 put it +45px below the chin;
// 0.22 puts it at +5). Larger rise = anchor higher on the cloth = garment
// renders higher.
const FIT_ANCHOR_RISE = 0.22;

// Measure a garment's CLOTH shoulder line from its bind-pose mesh.
// Several of these templates carry an SMPL-X rig whose shoulder JOINTS are
// far narrower than the cloth skinned to them, so the joints must not be
// used for size. The band is taken at the rig shoulder height (or a fixed
// fraction of the box height when the rig is missing); its x-extent is the
// garment's shoulder width and the band centre becomes the anchor the
// engine places on the user's tracked shoulder midpoint.
const measureGarment = (rawModel, modelBox, leftShoulder, rightShoulder) => {
    const positions = [];
    const v = new THREE.Vector3();
    rawModel.updateMatrixWorld(true);
    rawModel.traverse((node) => {
        if (node.isMesh && node.geometry?.attributes?.position) {
            const attr = node.geometry.attributes.position;
            for (let i = 0; i < attr.count; i++) {
                v.fromBufferAttribute(attr, i).applyMatrix4(node.matrixWorld);
                positions.push(v.x, v.y, v.z);
            }
        }
    });
    const box = modelBox;
    const size = box.getSize(new THREE.Vector3());
    let rigSpan = null;
    if (leftShoulder && rightShoulder) {
        rigSpan = leftShoulder.getWorldPosition(new THREE.Vector3())
            .distanceTo(rightShoulder.getWorldPosition(new THREE.Vector3()));
    }
    const fallback = () => {
        const center = box.getCenter(new THREE.Vector3());
        return { anchor: center, shoulderWidth: size.x * 0.75, rigSpan };
    };
    if (positions.length === 0) return fallback();

    const verts = new Float32Array(positions);
    const count = verts.length / 3;
    // Width-by-height profile of the bind-pose cloth. The CLOTH shoulder
    // line is the topmost row whose width reaches 60% of the garment's
    // widest row: above it only the hood/collar (narrow), below it the
    // shoulders and sleeves. Anchoring there keeps the seams on the user's
    // tracked joint line even when the rig skeleton is offset inside the
    // cloth (the baseballjacket template carries its joints ~10cm low).
    const buckets = 64;
    const yMin = box.min.y;
    const step = size.y / buckets;
    const widths = new Array(buckets).fill(0);
    const xLo = new Array(buckets).fill(Infinity);
    const xHi = new Array(buckets).fill(-Infinity);
    const zSum = new Array(buckets).fill(0);
    const zCnt = new Array(buckets).fill(0);
    for (let i = 0; i < count; i++) {
        const b = Math.min(buckets - 1, Math.max(0, Math.floor((verts[i * 3 + 1] - yMin) / step)));
        const x = verts[i * 3];
        widths[b]++;
        if (x < xLo[b]) xLo[b] = x;
        if (x > xHi[b]) xHi[b] = x;
        zSum[b] += verts[i * 3 + 2];
        zCnt[b]++;
    }
    let maxW = 0;
    for (let b = 0; b < buckets; b++) maxW = Math.max(maxW, xHi[b] - xLo[b]);
    let bandBucket = -1;
    for (let b = 0; b < buckets; b++) {
        if (widths[b] >= 32 && (xHi[b] - xLo[b]) >= 0.6 * maxW) { bandBucket = b; break; }
    }
    if (bandBucket < 0 || !Number.isFinite(xLo[bandBucket])) return fallback();
    const bandY = yMin + (bandBucket + 0.5) * step;
    const xMin = xLo[bandBucket];
    const xMax = xHi[bandBucket];
    const sumZ = zSum[bandBucket];
    const bandCount = zCnt[bandBucket];
    return {
        anchor: new THREE.Vector3((xMin + xMax) / 2, bandY, sumZ / bandCount),
        shoulderWidth: xMax - xMin,
        rigSpan,
    };
};

const disposeModel = (model) => {
    if (!model) return;
    model.traverse((child) => {
        if (child.isMesh) {
            child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => {
                        if (material.map) material.map.dispose();
                        material.dispose();
                    });
                } else {
                    if (child.material.map) child.material.map.dispose();
                    child.material.dispose();
                }
            }
        }
    });
};
const VtoCanvas = forwardRef(({ onMeshInfoUpdate, isAccessoryCategory, garmentModelPath, retailFiles }, ref) => {
    const { state } = useStateManager();
    const { isMobileLayout } = useDeviceDetection();
    const { poseLandmarks, poseWorldLandmarks, faceLandmarks, canvasDimensions, physicsEnabled } = state.viewState;
    const { selectedGarment, selectedGender } = state.vtoState;
    const { garmentData } = state.data;
    const garmentDetails = garmentData?.[selectedGender]?.[selectedGarment];
    const mountRef = useRef(null);
    const stateRef = useRef({}).current;

    const updateRetail = (model, files) => {
        const version = (stateRef.retailVersion || 0) + 1;
        stateRef.retailVersion = version;
        if (!model) return;
        if (!files?.front) {
            clearRetailUv(model);
            disposeRetailImages(stateRef.retailImages);
            stateRef.retailImages = null;
            return;
        }
        loadRetailImages(files.front, files.back).then(images => {
            if (version !== stateRef.retailVersion || model !== stateRef.garmentModel) {
                disposeRetailImages(images);
                return;
            }
            applyRetailUv(model, images);
            disposeRetailImages(stateRef.retailImages);
            stateRef.retailImages = images;
        }).catch(error => console.error('Retail image mapping failed:', error));
    };

    useEffect(() => {
        stateRef.retailFiles = retailFiles;
        if (!isAccessoryCategory) updateRetail(stateRef.garmentModel, retailFiles);
    }, [retailFiles, isAccessoryCategory]);

    useImperativeHandle(ref, () => ({
        takeSelfie: async () => {
            const backgroundCanvas = document.getElementById('background-canvas');
            const threeCanvas = stateRef.renderer?.domElement;

            if (!backgroundCanvas || !threeCanvas) {
                throw new Error("Selfie failed: Missing canvas elements");
            }

            const compositeCanvas = document.createElement('canvas');
            const scale = 2;
            compositeCanvas.width = canvasDimensions.width * scale;
            compositeCanvas.height = canvasDimensions.height * scale;
            const compositeCtx = compositeCanvas.getContext('2d');

            compositeCtx.imageSmoothingEnabled = true;
            compositeCtx.imageSmoothingQuality = 'high';

            compositeCtx.save();
            compositeCtx.scale(-1, 1);
            compositeCtx.translate(-compositeCanvas.width, 0);
            compositeCtx.drawImage(backgroundCanvas, 0, 0, compositeCanvas.width, compositeCanvas.height);
            compositeCtx.restore();

            compositeCtx.drawImage(threeCanvas, 0, 0, compositeCanvas.width, compositeCanvas.height);

            compositeCtx.font = `${12 * scale}px SF Pro Display`;
            compositeCtx.fillStyle = 'rgba(255, 255, 255, 0.6)';
            compositeCtx.textAlign = 'right';
            compositeCtx.fillText('softWEAR', compositeCanvas.width - (10 * scale), compositeCanvas.height - (10 * scale));

            return new Promise((resolve, reject) => {
                compositeCanvas.toBlob((blob) => {
                    if (blob) {
                        resolve(blob);
                    } else {
                        reject(new Error('Failed to create image blob'));
                    }
                }, 'image/png', 0.95);
            });
        }
    }));

    useEffect(() => {
        stateRef.poseLandmarks = poseLandmarks;
        stateRef.poseWorldLandmarks = poseWorldLandmarks;
        stateRef.faceLandmarks = faceLandmarks;
        stateRef.isAccessoryCategory = isAccessoryCategory;
        stateRef.garmentDetails = garmentDetails;
    }, [poseLandmarks, poseWorldLandmarks, faceLandmarks, isAccessoryCategory, garmentDetails]);

    useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        stateRef.scene = new THREE.Scene();
        stateRef.camera = new THREE.PerspectiveCamera(50, mount.offsetWidth / mount.offsetHeight, 0.1, 1000);
        stateRef.camera.position.set(0, 0.0, 2.5);
        stateRef.camera.lookAt(0, 0.0, 0);

        stateRef.renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true,
            preserveDrawingBuffer: true,
            powerPreference: "high-performance"
        });
        stateRef.renderer.setSize(mount.offsetWidth, mount.offsetHeight);
        stateRef.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        stateRef.renderer.setClearColor(0x000000, 0);
        stateRef.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        stateRef.renderer.outputColorSpace = THREE.SRGBColorSpace;
        mount.appendChild(stateRef.renderer.domElement);

        const handleResize = () => {
            if (mount && stateRef.renderer && stateRef.camera) {
                stateRef.camera.aspect = mount.offsetWidth / mount.offsetHeight;
                stateRef.camera.updateProjectionMatrix();
                stateRef.renderer.setSize(mount.offsetWidth, mount.offsetHeight);
            }
        };
        window.addEventListener('resize', handleResize);

        const ambientLight = new THREE.AmbientLight(0xffffff, 1);
        stateRef.scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
        keyLight.position.set(0.5, 1, 2);
        stateRef.scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0xffffff, 0.6);
        fillLight.position.set(-0.5, 0.5, 2);
        stateRef.scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xffffff, 0.7);
        rimLight.position.set(0, 1, -2);
        stateRef.scene.add(rimLight);

        // No negative scale here — mirroring is done at the DOM layer (CSS
        // scaleX(-1) on the canvas) so the 3D scene stays right-handed and
        // quaternion-driven bones rotate correctly.
        const worldGroup = new THREE.Group();
        stateRef.scene.add(worldGroup);
        stateRef.worldGroup = worldGroup;

        stateRef.poseEngine = new VtoPoseEngine();
        stateRef.headPoseMapper = new HeadPoseMapper(smplxBoneData);
        stateRef.poseMapper = new SMPLXPoseMapper({}, window.innerWidth <= 768);
        stateRef.physics = new GarmentPhysics();

        let lastTime = performance.now();
        let frameCount = 0;
        const animate = () => {
            stateRef.animFrameId = requestAnimationFrame(animate);
            const currentTime = performance.now();
            frameCount++;

            if (currentTime - lastTime >= 1000) {
                window.softWearPerformance.garmentFPS = frameCount;
                frameCount = 0;
                lastTime = currentTime;
            }

            const { poseEngine, headPoseMapper, poseMapper, renderer, scene, camera, garmentModel } = stateRef;

            if (garmentModel && renderer && scene && camera) {
                if (stateRef.isAccessoryCategory) {
                    const headTransform = headPoseMapper.update(stateRef.faceLandmarks, stateRef.garmentDetails, camera);
                    if (headTransform) {
                        garmentModel.position.lerp(headTransform.position, 0.1);
                        garmentModel.quaternion.slerp(headTransform.quaternion, 0.1);
                        garmentModel.scale.copy(headTransform.scale);
                    }
                } else {
                    // Prefer metric world landmarks; fall back to image landmarks
                    // (Holistic often returns no world landmarks). The body-frame
                    // math is scale-relative, so either works for orientation/aim.
                    // softWearPerformance.testWorldLandmarks (harness-only) wins
                    // so the headless fit test can exercise arm articulation.
                    const pose3d = (window.softWearPerformance && window.softWearPerformance.testWorldLandmarks)
                        || stateRef.poseWorldLandmarks || stateRef.poseLandmarks;
                    poseEngine.update(stateRef.poseLandmarks, pose3d, garmentModel, camera);
                    poseMapper.applyPoseToRiggedGarment(garmentModel, pose3d, currentTime);
                }
            }

            if (renderer && scene && camera) {
                renderer.render(scene, camera);
            }
        };
        animate();

        return () => {
            window.removeEventListener('resize', handleResize);
            if (stateRef.animFrameId) {
                cancelAnimationFrame(stateRef.animFrameId);
            }
            disposeModel(stateRef.garmentModel);
            if (mount && stateRef.renderer && mount.contains(stateRef.renderer.domElement)) {
                mount.removeChild(stateRef.renderer.domElement);
            }
            if (stateRef.renderer) {
                stateRef.renderer.dispose();
            }
        };
    }, []);

    useEffect(() => {
        if (!garmentModelPath) return;

        let isCancelled = false;
        let dracoLoader = null;
        let loader = null;

        const loadGarment = async () => {
            dracoLoader = new DRACOLoader();
            dracoLoader.setDecoderPath('./draco/');
            loader = new GLTFLoader();
            loader.setDRACOLoader(dracoLoader);

            const oldModel = stateRef.garmentModel;
            if (oldModel && stateRef.worldGroup) {
                stateRef.retailVersion = (stateRef.retailVersion || 0) + 1;
                clearRetailUv(oldModel);
                disposeRetailImages(stateRef.retailImages);
                stateRef.retailImages = null;
                stateRef.worldGroup.remove(oldModel);
                disposeModel(oldModel);
                stateRef.garmentModel = null;
                if (onMeshInfoUpdate) {
                    onMeshInfoUpdate({ vertices: 0, fileSize: 0 });
                }
            }

            let fileSize = 0;
            let finalPath = garmentModelPath;

            try {
                const garmentObject = { modelPath: garmentModelPath };
                finalPath = await resolveModelPathWithFallback(garmentObject, isMobileLayout);

                if (!finalPath) {
                    console.warn('No valid model path found, skipping garment load');
                    return;
                }

                const response = await fetch(finalPath, { method: 'HEAD' });
                if (response.ok) {
                    const contentLength = response.headers.get('content-length');
                    if (contentLength) {
                        fileSize = Math.round(parseInt(contentLength, 10) / 1024);
                    }
                }
            } catch (e) {
                console.warn('Could not fetch file size for model via HEAD request.');
            }

            loader.load(
                finalPath,
                (gltf) => {
                    if (isCancelled) return;

                    const rawModel = gltf.scene;
                    const modelWrapper = new THREE.Group();
                    modelWrapper.name = 'GarmentWrapper';

                    modelWrapper.add(rawModel);
                    const modelBox = new THREE.Box3().setFromObject(rawModel);
                    stateRef.headPoseMapper.setModelWidth(modelBox.getSize(new THREE.Vector3()).x);

                    // Anchor the garment to its measured cloth shoulder line
                    // and size it from that measurement (see measureGarment).
                    let leftShoulder = null;
                    let rightShoulder = null;
                    rawModel.traverse((node) => {
                        const name = node.name.toLowerCase().replace(/[^a-z0-9]/g, '');
                        if (name === 'leftshoulder') leftShoulder = node;
                        if (name === 'rightshoulder') rightShoulder = node;
                    });
                    const measurements = measureGarment(rawModel, modelBox, leftShoulder, rightShoulder);
                    // Scale from the rig joint distance (matches the tracked
                    // anatomical joints); cloth band centre anchors vertically.
                    stateRef.poseEngine.measuredRigSpan = measurements.rigSpan;
                    stateRef.poseEngine.measuredClothWidth = measurements.shoulderWidth;
                    stateRef.poseEngine.shoulderSpan = measurements.rigSpan
                        ? measurements.rigSpan * RIG_SPAN_FUDGE
                        : measurements.shoulderWidth / 1.3;
                    measurements.anchor.y -= FIT_ANCHOR_RISE * (modelBox.max.y - modelBox.min.y);
                    rawModel.position.sub(measurements.anchor);
                    stateRef.poseEngine.hasState = false;
                    stateRef.poseEngine.smoothDepth = 0;

                    modelWrapper.traverse((node) => {
                        if (node.isSkinnedMesh && node.material) {
                            node.material.skinning = true;
                        }
                    });

                    stateRef.garmentModel = modelWrapper;
                    if (stateRef.worldGroup) {
                        stateRef.worldGroup.add(modelWrapper);
                    }

                    if (stateRef.poseMapper) {
                        stateRef.poseMapper.initializeBones(modelWrapper);
                    }
                    if (!stateRef.isAccessoryCategory && stateRef.retailFiles?.front) {
                        updateRetail(modelWrapper, stateRef.retailFiles);
                    }

                    if (onMeshInfoUpdate) {
                        let vertexCount = 0;
                        modelWrapper.traverse(child => {
                            if (child.isMesh) {
                                vertexCount += child.geometry.attributes.position.count;
                            }
                        });
                        onMeshInfoUpdate({ vertices: vertexCount, fileSize: fileSize });
                    }
                },
                undefined,
                (error) => {
                    if (!isCancelled) {
                        console.error('ERROR: Failed to load 3D model:', error);
                        console.error('Attempted path:', finalPath);

                        // Cleanup DRACO loader on error
                        if (dracoLoader) {
                            dracoLoader.dispose();
                            dracoLoader = null;
                        }

                        if (onMeshInfoUpdate) {
                            onMeshInfoUpdate({ vertices: 0, fileSize: 0, error: 'Model load failed' });
                        }
                    }
                }
            );
        };

        loadGarment();

        return () => {
            isCancelled = true;
            if (dracoLoader) {
                dracoLoader.dispose();
            }
        };
    }, [garmentModelPath, onMeshInfoUpdate, isMobileLayout]);

    return <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 5, transform: 'scaleX(-1)' }} />;
});
export default VtoCanvas;
