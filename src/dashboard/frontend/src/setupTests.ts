import '@testing-library/jest-dom';
import 'whatwg-fetch';
import { vi } from 'vitest';

// Mock Plotly during unit tests to avoid requiring a full WebGL context.
vi.mock('plotly.js-dist-min', () => ({
  __esModule: true,
  default: {
    newPlot: vi.fn(),
    purge: vi.fn()
  }
}));
