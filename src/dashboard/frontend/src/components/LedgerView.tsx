import React, { useEffect, useState } from 'react';

type LedgerBlock = {
  index: number;
  hash: string;
  prev_hash?: string;
  projection?: number[];
  transactions?: Array<Record<string, unknown>>;
};

type LedgerResponse = {
  chain: LedgerBlock[];
  pending: Array<Record<string, unknown>>;
};

const LedgerView: React.FC = () => {
  const [data, setData] = useState<LedgerResponse>({ chain: [], pending: [] });
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/dashboard/ledger');
        if (!response.ok) {
          throw new Error(`Failed to load ledger: ${response.statusText}`);
        }
        const payload = (await response.json()) as LedgerResponse;
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

  return (
    <section aria-labelledby="ledger-heading">
      <h2 id="ledger-heading">Ledger</h2>
      {loading && <p>Loading ledger…</p>}
      {error && <p className="error">{error}</p>}
      {!loading && !error && (
        <>
          <p>
            Pending transactions: <strong>{data.pending.length}</strong>
          </p>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Hash</th>
                  <th>Previous</th>
                  <th>Projection (x₁, x₂, x₃)</th>
                </tr>
              </thead>
              <tbody>
                {data.chain.length === 0 && (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', opacity: 0.7 }}>
                      No blocks created yet. Interact with the API to populate the chain.
                    </td>
                  </tr>
                )}
                {data.chain.map((block) => (
                  <tr key={block.hash}>
                    <td>{block.index}</td>
                    <td>{block.hash.slice(0, 12)}…</td>
                    <td>{block.prev_hash ? `${block.prev_hash.slice(0, 12)}…` : 'genesis'}</td>
                    <td>
                      {block.projection?.length ? (
                        block.projection
                          .slice(0, 3)
                          .map((value) => value.toFixed(3))
                          .join(', ')
                      ) : (
                        <span style={{ opacity: 0.7 }}>n/a</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
};

export default LedgerView;
