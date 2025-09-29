import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

type MlResponse = {
  epochs: number[];
  timeline: string[];
  accuracy: number[];
  loss: number[];
};

const MlView: React.FC = () => {
  const [data, setData] = useState<MlResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/dashboard/ml/train_status');
        if (!response.ok) {
          throw new Error(`Failed to load ML status: ${response.statusText}`);
        }
        const payload = (await response.json()) as MlResponse;
        if (!cancelled) {
          setData(payload);
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
    if (!plotRef.current || !data) {
      return;
    }

    const accuracyTrace = {
      x: data.epochs,
      y: data.accuracy,
      mode: 'lines+markers',
      name: 'Accuracy',
      line: { color: '#7dd3fc' }
    };

    const lossTrace = {
      x: data.epochs,
      y: data.loss,
      mode: 'lines+markers',
      name: 'Loss',
      yaxis: 'y2',
      line: { color: '#f87171' }
    };

    const layout = {
      title: 'Demo training metrics',
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      xaxis: { title: 'Epoch', color: '#f4f6fb' },
      yaxis: { title: 'Accuracy', color: '#7dd3fc', rangemode: 'tozero' as const },
      yaxis2: {
        title: 'Loss',
        overlaying: 'y' as const,
        side: 'right' as const,
        color: '#f87171',
        rangemode: 'tozero' as const
      },
      legend: { orientation: 'h' as const },
      margin: { l: 50, r: 50, t: 40, b: 40 }
    };

    Plotly.newPlot(plotRef.current, [accuracyTrace, lossTrace], layout, { responsive: true });

    return () => {
      Plotly.purge(plotRef.current as HTMLDivElement);
    };
  }, [data]);

  return (
    <section aria-labelledby="ml-heading">
      <h2 id="ml-heading">ML Training</h2>
      {loading && <p>Loading training curve…</p>}
      {error && <p className="error">{error}</p>}
      <div ref={plotRef} style={{ width: '100%', height: '320px' }} />
    </section>
  );
};

export default MlView;
