import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import App from '../../App';

const jsonResponse = (data: unknown) =>
  Promise.resolve(
    new Response(JSON.stringify(data), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    })
  );

describe('App dashboard shell', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    global.fetch = vi.fn((input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.startsWith('/dashboard/ledger')) {
        return jsonResponse({ chain: [], pending: [] });
      }
      if (url.startsWith('/dashboard/hdag')) {
        return jsonResponse({ nodes: [], edges: [] });
      }
      if (url.startsWith('/dashboard/spiral')) {
        return jsonResponse({
          points: [
            { theta: 0, coordinates: [0, 0, 0, 0, 0] },
            { theta: 1, coordinates: [1, 1, 0.5, 0.5, 0.1] }
          ],
          params: { a: 1, b: 0.5, c: 0.1 }
        });
      }
      if (url.startsWith('/dashboard/tic')) {
        return jsonResponse({ state: [0.1, 0.2, 0.3], history: [[0, 0, 0], [0.1, 0.2, 0.3]] });
      }
      if (url.startsWith('/dashboard/ml/train_status')) {
        return jsonResponse({
          epochs: [1, 2],
          timeline: [],
          accuracy: [0.5, 0.6],
          loss: [1.0, 0.8]
        });
      }
      if (url.startsWith('/dashboard/zkml/proof')) {
        return jsonResponse({
          input: [1, 2, 3],
          prediction: 0.75,
          proof: 'proof',
          statement: 'statement',
          verified: true
        });
      }
      if (url.startsWith('/zkml/verify')) {
        return jsonResponse({ valid: true });
      }
      return jsonResponse({});
    }) as typeof fetch;
  });

  it('renders all dashboard sections', async () => {
    render(<App />);

    await waitFor(() => expect(screen.getByText(/Rings of Saturn Dashboard/i)).toBeInTheDocument());
    expect(await screen.findByText(/Ledger/i)).toBeInTheDocument();
    expect(await screen.findByText(/HDAG/i)).toBeInTheDocument();
    expect(await screen.findByText(/Spiral/i)).toBeInTheDocument();
    expect(await screen.findByText(/TIC Condensate/i)).toBeInTheDocument();
    expect(await screen.findByText(/ML Training/i)).toBeInTheDocument();
    expect(await screen.findByText(/ZKML Inference/i)).toBeInTheDocument();
  });
});
