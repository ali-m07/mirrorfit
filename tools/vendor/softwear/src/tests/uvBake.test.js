// src/tests/uvBake.test.js
import { solveAffine, triangleFacing, classifyMaterial } from '../../../../retail3d/uvBake';

describe('uvBake pure helpers', () => {
    describe('solveAffine', () => {
        test('maps the three src points exactly onto the three dst points', () => {
            const src = [[10, 10], [60, 12], [18, 58]];
            const dst = [[0, 0], [100, 0], [0, 100]];
            const [a, b, c, d, e, f] = solveAffine(src, dst);
            src.forEach(([x, y], i) => {
                expect(a * x + c * y + e).toBeCloseTo(dst[i][0], 6);
                expect(b * x + d * y + f).toBeCloseTo(dst[i][1], 6);
            });
        });

        test('handles rotation, scale and translation combined', () => {
            // 90° rotation about the origin scaled by 2, translated by (5, -3)
            const src = [[1, 0], [3, 1], [0, 4]];
            const expected = src.map(([x, y]) => [-2 * y + 5, 2 * x - 3]);
            const [a, b, c, d, e, f] = solveAffine(src, expected);
            src.forEach(([x, y], i) => {
                expect(a * x + c * y + e).toBeCloseTo(expected[i][0], 6);
                expect(b * x + d * y + f).toBeCloseTo(expected[i][1], 6);
            });
        });

        test('throws on collinear src points', () => {
            expect(() => solveAffine(
                [[0, 0], [1, 1], [2, 2]], [[0, 0], [10, 0], [0, 10]])).toThrow();
        });

        test('throws on collinear dst points', () => {
            expect(() => solveAffine(
                [[0, 0], [10, 0], [0, 10]], [[0, 0], [1, 1], [2, 2]])).toThrow();
        });
    });

    describe('triangleFacing', () => {
        test('returns front for CCW winding with +Z normal', () => {
            expect(triangleFacing([0, 0], [1, 0], [0, 1])).toBe('front');
        });

        test('returns back for the mirrored winding', () => {
            expect(triangleFacing([0, 0], [0, 1], [1, 0])).toBe('back');
        });

        test('is stable under translation', () => {
            expect(triangleFacing([100, 50], [101, 50], [100, 51])).toBe('front');
            expect(triangleFacing([100, 50], [100, 51], [101, 50])).toBe('back');
        });
    });

    describe('classifyMaterial', () => {
        test('routes CLO3D panel materials by name', () => {
            expect(classifyMaterial('MAIN FABRIC_FRONT_1946.001')).toBe('front');
            expect(classifyMaterial('ARMZ_BACK_1961.001')).toBe('back');
            expect(classifyMaterial('SLEEVE HEMZ_BACK_1973.001')).toBe('back');
        });

        test('leaves SIDE panels and unnamed trims untouched', () => {
            expect(classifyMaterial('Leather_Lambskin_SIDE_84543')).toBe('keep');
            expect(classifyMaterial('Material3743')).toBe('keep');
        });
    });
});
