let THREE;
export function configureRetailThree(three) { THREE = three; }

// Clothware's useful separation is kept here: reconstruct a retail image once
// onto a garment template, then let the existing skeleton drive that template
// in the live renderer. This is a projection onto a fixed mesh, not inferred
// garment geometry or physical fit prediction.

async function imageTexture(file) {
  const bitmap = await createImageBitmap(file);
  const ratio = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
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

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  return texture;
}

export async function loadRetailTextures(frontFile, backFile) {
  if (!frontFile) return null;
  const front = await imageTexture(frontFile);
  try {
    const back = backFile ? await imageTexture(backFile) : front;
    return { front, back };
  } catch (error) {
    front.dispose();
    throw error;
  }
}

export function disposeRetailTextures(textures) {
  if (!textures) return;
  textures.front.dispose();
  if (textures.back !== textures.front) textures.back.dispose();
}

export function applyRetailTextures(model, textures) {
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  model.traverse(node => {
    if (!node.isMesh) return;
    const positions = node.geometry?.getAttribute('position');
    if (!positions) return;
    for (let i = 0; i < positions.count; i++) {
      bounds.minX = Math.min(bounds.minX, positions.getX(i));
      bounds.maxX = Math.max(bounds.maxX, positions.getX(i));
      bounds.minY = Math.min(bounds.minY, positions.getY(i));
      bounds.maxY = Math.max(bounds.maxY, positions.getY(i));
    }
  });
  if (!Number.isFinite(bounds.minX) || bounds.maxX - bounds.minX < 1e-4 ||
      bounds.maxY - bounds.minY < 1e-4) throw new Error('Template has no usable garment geometry');

  model.traverse(node => {
    if (!node.isMesh) return;
    if (!node.userData.retailOriginalMaterial) node.userData.retailOriginalMaterial = node.material;
    else if (node.material !== node.userData.retailOriginalMaterial) {
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      materials.forEach(material => material.dispose());
    }
    const base = node.userData.retailOriginalMaterial;
    const originals = Array.isArray(base) ? base : [base];
    const materials = originals.map(() => {
      const material = new THREE.MeshStandardMaterial({
        color: 0xffffff, side: THREE.DoubleSide, roughness: 0.95,
        transparent: true, alphaTest: 0.04, depthWrite: true
      });
      material.onBeforeCompile = shader => {
        shader.uniforms.retailFront = { value: textures.front };
        shader.uniforms.retailBack = { value: textures.back };
        shader.uniforms.retailBounds = { value: new THREE.Vector4(
          bounds.minX, bounds.minY, bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) };
        shader.vertexShader = shader.vertexShader.replace('#include <common>',
          '#include <common>\nvarying vec2 vRetailXY;\nvarying float vRetailFacing;');
        shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
          '#include <begin_vertex>\nvRetailXY = position.xy;\nvRetailFacing = normal.z;');
        shader.fragmentShader = shader.fragmentShader.replace('#include <common>',
          '#include <common>\nuniform sampler2D retailFront;\nuniform sampler2D retailBack;\nuniform vec4 retailBounds;\nvarying vec2 vRetailXY;\nvarying float vRetailFacing;');
        shader.fragmentShader = shader.fragmentShader.replace('#include <map_fragment>',
          'vec2 retailUV = clamp((vRetailXY - retailBounds.xy) / retailBounds.zw, 0.0, 1.0);\n' +
          'vec4 retailPixel = vRetailFacing >= 0.0 ? texture2D(retailFront, retailUV) : texture2D(retailBack, vec2(1.0 - retailUV.x, retailUV.y));\n' +
          'diffuseColor.rgb = retailPixel.rgb;\ndiffuseColor.a *= retailPixel.a;');
      };
      material.customProgramCacheKey = () => 'mirrorfit-retail-projection-v1';
      return material;
    });
    node.material = Array.isArray(base) ? materials : materials[0];
  });
}

export function clearRetailTextures(model) {
  model?.traverse(node => {
    if (!node.isMesh || !node.userData.retailOriginalMaterial) return;
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    materials.forEach(material => material.dispose());
    node.material = node.userData.retailOriginalMaterial;
    delete node.userData.retailOriginalMaterial;
  });
}
