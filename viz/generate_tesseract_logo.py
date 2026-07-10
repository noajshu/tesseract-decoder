import math
import random

random.seed(0)

BASE_VERTICES = [
    (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
    (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
]

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


def rotate_point(point, angles):
    ax, ay, az = angles
    x, y, z = point
    cosx, sinx = math.cos(ax), math.sin(ax)
    cosy, siny = math.cos(ay), math.sin(ay)
    cosz, sinz = math.cos(az), math.sin(az)

    y, z = y * cosx - z * sinx, y * sinx + z * cosx
    x, z = x * cosy + z * siny, -x * siny + z * cosy
    x, y = x * cosz - y * sinz, x * sinz + y * cosz
    return (x, y, z)


def project_point(point):
    x, y, z = point
    return (x - 0.5 * z, y - 0.5 * z)


def generate_cube(angles, scale=1.0):
    """Return cube vertices rotated by given angles and scaled."""
    rotated = [rotate_point(v, angles) for v in BASE_VERTICES]
    return [(x * scale, y * scale, z * scale) for x, y, z in rotated]


def project_scale(cube1, cube2):
    """Project both cubes to 2D using a shared scaling for relative size."""
    p1 = [project_point(v) for v in cube1]
    p2 = [project_point(v) for v in cube2]

    xs = [pt[0] for pt in p1 + p2]
    ys = [pt[1] for pt in p1 + p2]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = 80 / max(max_x - min_x, max_y - min_y)

    def apply(points):
        out = []
        for x, y in points:
            px = (x - min_x) * scale + 10
            py = (y - min_y) * scale + 10
            out.append((px, py))
        return out

    return apply(p1), apply(p2)


def svg_lines(points1, points2):
    lines = []
    for a, b in EDGES:
        x1, y1 = points1[a]
        x2, y2 = points1[b]
        lines.append((x1, y1, x2, y2))
    for a, b in EDGES:
        x1, y1 = points2[a]
        x2, y2 = points2[b]
        lines.append((x1, y1, x2, y2))
    for i in range(8):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        lines.append((x1, y1, x2, y2))
    return lines


def main(out_path="docs/images/tesseract-logo.svg"):
    angles_outer = [random.uniform(-0.35, 0.35) for _ in range(3)]
    angles_inner = [random.uniform(-0.35, 0.35) for _ in range(3)]

    outer = generate_cube(angles_outer, scale=1.0)
    inner = generate_cube(angles_inner, scale=0.5)

    proj_outer, proj_inner = project_scale(outer, inner)
    lines = svg_lines(proj_outer, proj_inner)

    with open(out_path, "w") as f:
        f.write("<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160' viewBox='0 0 100 100'>\n")
        f.write("<style>.edge{stroke:#5f6368;stroke-width:2;fill:none}</style>\n")
        for x1, y1, x2, y2 in lines:
            f.write(f"  <line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' class='edge'/>\n")
        f.write("  <text x='50' y='95' fill='#5f6368' font-family='sans-serif' font-size='10' text-anchor='middle'>Tesseract</text>\n")
        f.write("</svg>\n")


if __name__ == "__main__":
    main()
