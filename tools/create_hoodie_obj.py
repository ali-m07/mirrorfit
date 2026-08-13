"""Create a lightweight, neutral hoodie mesh for local 3D pipeline tests."""
from pathlib import Path

OUT = Path(__file__).parents[1] / "assets" / "clothes" / "basic_hoodie.obj"
verts = []
faces = []

def box(cx, cy, cz, sx, sy, sz):
    base = len(verts) + 1
    for x, y, z in ((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                    (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)):
        verts.append((cx+x*sx/2, cy+y*sy/2, cz+z*sz/2))
    faces.extend([(base,base+1,base+2,base+3),(base+4,base+7,base+6,base+5),
                  (base,base+4,base+5,base+1),(base+1,base+5,base+6,base+2),
                  (base+2,base+6,base+7,base+3),(base+4,base,base+3,base+7)])

# Coordinate system: X width, Y height, Z depth.
box(0, 1.25, 0, 2.8, 3.0, 0.75)       # torso
box(-1.85, 1.35, 0, 0.75, 2.7, 0.72)  # left sleeve
box(1.85, 1.35, 0, 0.75, 2.7, 0.72)   # right sleeve
box(0, 3.05, -0.02, 1.65, 1.25, 0.9)  # hood
box(0, 0.45, -0.42, 1.35, 0.42, 0.12) # front pocket

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    f.write("# MirrorFit basic hoodie test mesh\n")
    for v in verts: f.write("v %.5f %.5f %.5f\n" % v)
    for face in faces: f.write("f %s\n" % " ".join(map(str, face)))
print(OUT)
