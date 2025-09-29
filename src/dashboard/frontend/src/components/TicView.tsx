import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

type TicResponse = {
  state: number[] | null;
  history: number[][];
};

const TicView: React.FC = () => {
  const [state, setState] = useState<number[] | null>(null);
  const [history, setHistory] = useState<number[][]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/dashboard/tic');
        if (!response.ok) {
          throw new Error(`Failed to load TIC state: ${response.statusText}`);
        }
        const payload = (await response.json()) as TicResponse;
        if (!cancelled) {
          setState(payload.state ?? null);
          setHistory(payload.history ?? []);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!plotRef.current || history.length === 0) {
      return;
    }

    const xs = history.map((vec) => vec[0] ?? 0);
    const ys = history.map((vec) => vec[1] ?? 0);
    const zs = history.map((vec) => vec[2] ?? 0);

    const trace = {
      x: xs,
      y: ys,
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: 10,
        color: zs,
        colorscale: 'Portland',
        showscale: true
      },
      text: history.map((vec, idx) => `v${idx}: ${vec.map((v) => v.toFixed(2)).join(', ')}`)
    };

    const highlight =
      state && state.length >= 2
        ? [{
            x: [state[0]],
            y: [state[1]],
            mode: 'markers',
            type: 'scatter',
            marker: { size: 14, color: '#ffdd57', symbol: 'star' },
            name: 'Condensate'
          }]
        : [];

    const layout = {
      title: 'TIC condensed vectors',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: { title: 'Component 1', color: '#f4f6fb' },
      yaxis: { title: 'Component 2', color: '#f4f6fb' },
      margin: { l: 40, r: 10, b: 40, t: 40 }
    };

    Plotly.newPlot(plotRef.current, [trace, ...highlight], layout, { responsive: true });

    return () => {
      Plotly.purge(plotRef.current as HTMLDivElement);
    };
  }, [history, state]);

  return (
    <section aria-labelledby="tic-heading">
      <h2 id="tic-heading">TIC Condensate</h2>
      {loading && <p>Loading TIC vectors…</p>}
      {error && <p className="error">{error}</p>}
      {state && (
        <p>
          Current condensate:{' '}
          <code>{state.map((value) => value.toFixed(3)).join(', ')}</code>
        </p>
      )}
      <div ref={plotRef} style={{ width: '100%', height: '360px' }} />
    </section>
  );
};

export default TicView;
