import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

type SpiralParams = {
  a: number;
  b: number;
  c: number;
};

type SpiralPoint = {
  theta: number;
  coordinates: number[];
};

type SpiralResponse = {
  points: SpiralPoint[];
  params: SpiralParams;
};

const DEFAULT_PARAMS: SpiralParams = { a: 1, b: 0.5, c: 0.1 };

const SpiralView: React.FC = () => {
  const [params, setParams] = useState<SpiralParams>(DEFAULT_PARAMS);
  const [points, setPoints] = useState<SpiralPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      try {
        const query = new URLSearchParams({
          n: '200',
          a: params.a.toString(),
          b: params.b.toString(),
          c: params.c.toString()
        });
        const response = await fetch(`/dashboard/spiral?${query.toString()}`);
        if (!response.ok) {
          throw new Error(`Failed to load spiral: ${response.statusText}`);
        }
        const payload = (await response.json()) as SpiralResponse;
        if (!cancelled) {
          setPoints(payload.points);
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
  }, [params]);

  useEffect(() => {
    if (!plotRef.current || points.length === 0) {
      return;
    }
    const xs = points.map((point) => point.coordinates[0]);
    const ys = points.map((point) => point.coordinates[1]);
    const zs = points.map((point) => point.coordinates[2]);

    const trace = {
      x: xs,
      y: ys,
      z: zs,
      mode: 'lines+markers',
      type: 'scatter3d' as const,
      marker: { size: 3, color: points.map((p) => p.theta), colorscale: 'Viridis' },
      line: { width: 2, color: '#6a5acd' }
    };

    const layout = {
      autosize: true,
      title: 'Spiral Projection (3D)',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      scene: {
        xaxis: { title: 'x₁', color: '#f4f6fb' },
        yaxis: { title: 'x₂', color: '#f4f6fb' },
        zaxis: { title: 'x₃', color: '#f4f6fb' }
      },
      margin: { l: 0, r: 0, b: 0, t: 40 }
    };

    Plotly.newPlot(plotRef.current, [trace], layout, { responsive: true });

    return () => {
      Plotly.purge(plotRef.current as HTMLDivElement);
    };
  }, [points]);

  const handleParamChange = (key: keyof SpiralParams) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(event.target.value);
    setParams((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <section aria-labelledby="spiral-heading">
      <h2 id="spiral-heading">Spiral</h2>
      <div className="controls">
        <label>
          a: {params.a.toFixed(2)}
          <input type="range" min="0.2" max="2" step="0.05" value={params.a} onChange={handleParamChange('a')} />
        </label>
        <label>
          b: {params.b.toFixed(2)}
          <input type="range" min="0.1" max="1.5" step="0.05" value={params.b} onChange={handleParamChange('b')} />
        </label>
        <label>
          c: {params.c.toFixed(2)}
          <input type="range" min="0.05" max="0.5" step="0.01" value={params.c} onChange={handleParamChange('c')} />
        </label>
      </div>
      {loading && <p>Generating spiral…</p>}
      {error && <p className="error">{error}</p>}
      <div ref={plotRef} style={{ width: '100%', height: '420px' }} />
    </section>
  );
};

export default SpiralView;
