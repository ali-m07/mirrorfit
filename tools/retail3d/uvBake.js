let THREE;
export function configureUvBake(three) { THREE = three; }

// UV-atlas bake for retail product images. Replaces the screen-space
// projection shader in retailTexture.js: instead of projecting at draw time,
// rasterize each product triangle into the destination texture space of the
// garment's TEXCOORD_0 atlas, so the result is a plain texture the existing
// skinned renderer samples unchanged.
//
// Bind space: triangle positions are transformed by
//   bind = (model.matrixWorld^-1 * node.matrixWorld)
// captured once at call time, i.e. the mesh node's transform relative to the
// wrapper. The wrapper transform is inverted out of both matrices, so a live
// root animation present in both cancels; skeleton bone motion does not move
// the mesh node's own matrixWorld, so positions stay in the glTF bind pose.
// matrixWorld values must be current when this runs (call
// model.updateWorldMatrix(true, true) first if unsure) and the garment must
// face +Z in bind pose (CLO3D exports do).

const MAX_DIMENSION = 1024;

// Same background removal as retailTexture.js (corner-alpha check + flood
// fill from the borders), but the result stays a plain canvas: baking is
// synchronous canvas work and must not depend on THREE.
async function imageCanvas(file) {
  const bitmap = await createImageBitmap(file);
  const ratio = Math.min(1, MAX_DIMENSION / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * ratio));
  canvas.height = Math.max(1, Math.round(bitmap.height * ratio));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();

  const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const pixels = image.data;
  const width = canvas.width;
  const height = canvas.height;
  const corner = [0, (width - 1) * 4, (height - 1) * width * 4,
    ((height - 1) * width + width - 1) * 4];
  const hasAlpha = corner.some(i => pixels[i + 3] < 240);
  if (!hasAlpha) {
    const bg = [0, 1, 2].map(c => Math.round(corner.reduce((sum, i) => sum + pixels[i + c], 0) / 4));
    const seen = new Uint8Array(width * height);
    const queue = new Int32Array(width * height);
    let head = 0;
    let tail = 0;
    const add = (index) => {
      if (seen[index]) return;
      seen[index] = 1;
      const offset = index * 4;
      const distance = Math.max(...bg.map((value, c) => Math.abs(pixels[offset + c] - value)));
      if (distance < 32) queue[tail++] = index;
    };
    for (let x = 0; x < width; x++) { add(x); add((height - 1) * width + x); }
    for (let y = 0; y < height; y++) { add(y * width); add(y * width + width - 1); }
    while (head < tail) {
      const index = queue[head++];
      pixels[index * 4 + 3] = 0;
      const x = index % width;
      const y = Math.floor(index / width);
      if (x > 0) add(index - 1);
      if (x + 1 < width) add(index + 1);
      if (y > 0) add(index - width);
      if (y + 1 < height) add(index + width);
    }
    ctx.putImageData(image, 0, 0);
  }
  return { canvas, hasAlpha };
}

export async function loadRetailImages(frontFile, backFile) {
  if (!frontFile) return null;
  return {
    front: await imageCanvas(frontFile),
    back: backFile ? await imageCanvas(backFile) : null
  };
}

// Maps the src triangle onto the dst triangle: returns [a, b, c, d, e, f] for
// ctx.setTransform(a, b, c, d, e, f), where dst.x = a*x + c*y + e and
// dst.y = b*x + d*y + f. Throws on collinear (zero-area) input.
export function solveAffine(src, dst) {
  const ux = src[1][0] - src[0][0];
  const uy = src[1][1] - src[0][1];
  const vx = src[2][0] - src[0][0];
  const vy = src[2][1] - src[0][1];
  const px = dst[1][0] - dst[0][0];
  const py = dst[1][1] - dst[0][1];
  const qx = dst[2][0] - dst[0][0];
  const qy = dst[2][1] - dst[0][1];
  const det = ux * vy - uy * vx;
  if (!Number.isFinite(det) || Math.abs(det) < 1e-12 ||
      Math.abs(px * qy - py * qx) < 1e-12) throw new Error('Degenerate triangle');
  // dst edges = M * src edges with M = [[a, c], [b, d]]; solve the 2x2 system.
  const a = (px * vy - qx * uy) / det;
  const b = (py * vy - qy * uy) / det;
  const c = (qx * ux - px * vx) / det;
  const d = (qy * ux - py * vx) / det;
  const e = dst[0][0] - (a * src[0][0] + c * src[0][1]);
  const f = dst[0][1] - (b * src[0][0] + d * src[0][1]);
  return [a, b, c, d, e, f];
}

// Sign of the bind-space face normal z from the winding: CCW in XY (+Z
// normal) is front. Computed from positions, not the NORMAL attribute, so
// double-sided panels classify correctly.
export function triangleFacing(p0, p1, p2) {
  const z = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
  return z >= 0 ? 'front' : 'back';
}

// CLO3D splits garment panels into per-side materials. Names without a side
// hint (jacket trims like "Material3743", SIDE panels) keep their texture.
export function classifyMaterial(name) {
  if (/BACK/i.test(name)) return 'back';
  if (/FRONT/i.test(name)) return 'front';
  return 'keep';
}

function flatColorCanvas(material) {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = material.color ? material.color.getStyle() : '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return canvas;
}

// Clone the material's current texture image into a fresh canvas at the
// original resolution; fall back to a flat color canvas when there is no
// drawable map (compressed/data textures throw on drawImage).
function cloneTextureCanvas(material) {
  const image = material.map && material.map.image;
  if (image && image.width > 0 && image.height > 0) {
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    try {
      canvas.getContext('2d').drawImage(image, 0, 0);
      return canvas;
    } catch (error) {
      // Fall through to the flat-color canvas.
    }
  }
  return flatColorCanvas(material);
}

// Draw one source triangle (product-image pixels) into one destination
// triangle (atlas pixels) via a clipped affine transform. Both triangles are
// grown about their centroids by the same relative factor: an affine map
// commutes with that expansion, so the solve stays exact while the clip path
// gains ~1.5px of outward padding plus matching source bleed — enough to
// hide seams between adjacent triangles.
function drawProductTriangle(ctx, source, src, dst) {
  const centroid = tri => [
    (tri[0][0] + tri[1][0] + tri[2][0]) / 3,
    (tri[0][1] + tri[1][1] + tri[2][1]) / 3
  ];
  const [dcx, dcy] = centroid(dst);
  const radius = dst.reduce((sum, p) => sum + Math.hypot(p[0] - dcx, p[1] - dcy), 0) / 3;
  if (radius < 1e-6) return;
  const factor = 1 + 1.5 / radius;
  const grow = tri => {
    const [cx, cy] = centroid(tri);
    return tri.map(p => [cx + (p[0] - cx) * factor, cy + (p[1] - cy) * factor]);
  };
  const grownDst = grow(dst);
  let transform;
  try {
    transform = solveAffine(grow(src), grownDst);
  } catch (error) {
    return; // degenerate after growth — nothing sensible to draw
  }
  const [a, b, c, d, e, f] = transform;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(grownDst[0][0], grownDst[0][1]);
  ctx.lineTo(grownDst[1][0], grownDst[1][1]);
  ctx.lineTo(grownDst[2][0], grownDst[2][1]);
  ctx.closePath();
  ctx.clip();
  ctx.setTransform(a, b, c, d, e, f);
  // source-over: transparent product pixels keep the garment texture below.
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(source, 0, 0);
  ctx.restore();
}

function releaseBaked(node) {
  const materials = Array.isArray(node.material) ? node.material : [node.material];
  materials.forEach(material => {
    if (!material || !material.userData || !material.userData.uvBake) return;
    if (material.map && material.map.userData && material.map.userData.uvBake) {
      material.map.dispose();
    }
    material.dispose();
  });
}

// Restore original materials and dispose baked materials + textures. 'keep'
// slots hold the original material object itself and are never disposed.
function restoreOriginals(model) {
  model.traverse(node => {
    if (!node.isMesh || !node.userData.uvBakeOriginal) return;
    releaseBaked(node);
    node.material = node.userData.uvBakeOriginal;
    delete node.userData.uvBakeOriginal;
  });
}

export function applyRetailUv(model, images) {
  if (!THREE) throw new Error('configureUvBake(three) must be called before applyRetailUv');
  if (!images || !images.front) throw new Error('No product images provided');

  // Idempotency: re-applying always bakes from the original materials, never
  // on top of a previous bake (mirrors the retailTexture.js bookkeeping).
  restoreOriginals(model);
  model.updateWorldMatrix(true, true);

  const meshes = [];
  let anyUv = false;
  model.traverse(node => {
    if (!node.isMesh || !node.geometry) return;
    if (node.geometry.getAttribute('uv')) anyUv = true; // TEXCOORD_0
    meshes.push(node);
  });
  if (!anyUv) throw new Error('No UV coordinates found');

  const modelInverse = new THREE.Matrix4().copy(model.matrixWorld).invert();
  const bind = new THREE.Matrix4();
  const vertex = new THREE.Vector3();

  for (const node of meshes) {
    if (!node.material) continue;
    const geometry = node.geometry;
    const uv = geometry.getAttribute('uv');
    const positions = geometry.getAttribute('position');
    if (!uv || !positions) continue;
    const index = geometry.getIndex();
    const triangleCount = Math.floor((index ? index.count : positions.count) / 3);
    if (triangleCount < 1) continue;

    // Bind space, captured once per node (see header comment).
    bind.copy(modelInverse).multiply(node.matrixWorld);
    const count = positions.count;
    const bx = new Float32Array(count);
    const by = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      vertex.fromBufferAttribute(positions, i).applyMatrix4(bind);
      bx[i] = vertex.x;
      by[i] = vertex.y;
    }

    const isMultiMaterial = Array.isArray(node.material);
    const groups = isMultiMaterial && geometry.groups && geometry.groups.length
      ? geometry.groups
      : [{ start: 0, count: triangleCount * 3, materialIndex: 0 }];
    const originals = isMultiMaterial ? node.material : [node.material];
    const baked = originals.map(() => null);
    let changed = false;

    originals.forEach((material, slot) => {
      if (!material) return;
      const side = classifyMaterial(material.name || '');
      if (side === 'keep') return;
      // BACK falls back to the front image when no back photo was provided.
      const target = side === 'back' && images.back ? images.back : images.front;

      const tris = [];
      for (const group of groups) {
        if (isMultiMaterial && group.materialIndex !== slot) continue;
        for (let i = group.start; i + 2 < group.start + group.count; i += 3) {
          tris.push(index ? index.getX(i) : i,
            index ? index.getX(i + 1) : i + 1,
            index ? index.getX(i + 2) : i + 2);
        }
      }
      if (!tris.length) return;

      // Orthographic +Z projection over THIS material's own bind-space box.
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (let t = 0; t < tris.length; t++) {
        minX = Math.min(minX, bx[tris[t]]);
        maxX = Math.max(maxX, bx[tris[t]]);
        minY = Math.min(minY, by[tris[t]]);
        maxY = Math.max(maxY, by[tris[t]]);
      }
      const width = maxX - minX;
      const height = maxY - minY;
      if (width < 1e-6 || height < 1e-6) return;

      const dstCanvas = cloneTextureCanvas(material);
      const ctx = dstCanvas.getContext('2d');
      const sourceCanvas = target.canvas;

      for (let t = 0; t < tris.length; t += 3) {
        const ia = tris[t], ib = tris[t + 1], ic = tris[t + 2];
        // Degenerate UVs (all three identical) would collapse the clip.
        const ua = uv.getX(ia), va = uv.getY(ia);
        const ub = uv.getX(ib), vb = uv.getY(ib);
        const uc = uv.getX(ic), vc = uv.getY(ic);
        if (ua === ub && ua === uc && va === vb && va === vc) continue;

        // Mirror the horizontal source coordinate for back surfaces: the
        // whole back panel when routed to the back photo, and any reverse-
        // wound triangle of a double-sided panel (matches the facing test in
        // the old retailTexture shader).
        const facing = triangleFacing([bx[ia], by[ia]], [bx[ib], by[ib]], [bx[ic], by[ic]]);
        const flip = side === 'back' || facing === 'back';

        const dst = [
          [ua * dstCanvas.width, va * dstCanvas.height],
          [ub * dstCanvas.width, vb * dstCanvas.height],
          [uc * dstCanvas.width, vc * dstCanvas.height]
        ];
        const src = [ia, ib, ic].map(i => {
          let u = (bx[i] - minX) / width;
          if (flip) u = 1 - u;
          const v = 1 - (by[i] - minY) / height;
          return [u * sourceCanvas.width, v * sourceCanvas.height];
        });
        drawProductTriangle(ctx, sourceCanvas, src, dst);
      }

      const texture = new THREE.CanvasTexture(dstCanvas);
      texture.flipY = false; // glTF UV convention — critical
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.userData.uvBake = true;

      const bakedMaterial = material.clone();
      bakedMaterial.map = texture;
      bakedMaterial.userData.uvBake = true;
      bakedMaterial.needsUpdate = true; // recompile: map went from possibly null
      baked[slot] = bakedMaterial;
      changed = true;
    });

    if (changed) {
      node.userData.uvBakeOriginal = node.material;
      node.material = isMultiMaterial ? baked : baked[0];
    }
  }
}

export function clearRetailUv(model) {
  if (!model) return;
  restoreOriginals(model);
}

// Canvases are garbage collected like any DOM object; kept for API symmetry
// with disposeRetailTextures.
export function disposeRetailImages(images) {
  void images;
}
