import React, { useEffect, useState } from 'react';

type ProofPayload = {
  input: number[];
  prediction: number | null;
  proof: string;
  statement: string;
  verified: boolean;
};

const defaultProof: ProofPayload = {
  input: [],
  prediction: null,
  proof: '',
  statement: '',
  verified: false
};

const parseVector = (value: string): number[] => {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map((item) => Number.parseFloat(item))
    .filter((num) => !Number.isNaN(num));
};

const ZkmlView: React.FC = () => {
  const [vectorInput, setVectorInput] = useState<string>('1, 0.5, -0.5');
  const [proof, setProof] = useState<ProofPayload>(defaultProof);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadProof = async () => {
      try {
        const response = await fetch('/dashboard/zkml/proof');
        if (!response.ok) {
          throw new Error('Unable to retrieve previous proof');
        }
        const payload = (await response.json()) as ProofPayload;
        if (!cancelled) {
          setProof({ ...defaultProof, ...payload });
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Unknown error');
        }
      }
    };
    void loadProof();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setError(null);
    const vector = parseVector(vectorInput);
    if (vector.length === 0) {
      setError('Provide at least one numeric value.');
      return;
    }
    setLoading(true);
    try {
      const inferResponse = await fetch('/zkml/zk_infer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vector })
      });
      if (!inferResponse.ok) {
        throw new Error(`Inference failed: ${inferResponse.statusText}`);
      }
      const inferPayload = (await inferResponse.json()) as ProofPayload;
      let verified = false;
      if (inferPayload.statement && inferPayload.proof) {
        const verifyResponse = await fetch('/zkml/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ statement: inferPayload.statement, proof: inferPayload.proof })
        });
        if (verifyResponse.ok) {
          const verifyPayload = (await verifyResponse.json()) as { valid: boolean };
          verified = verifyPayload.valid;
        }
      }
      setProof({
        input: vector,
        prediction: inferPayload.prediction,
        proof: inferPayload.proof,
        statement: inferPayload.statement,
        verified
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section aria-labelledby="zkml-heading">
      <h2 id="zkml-heading">ZKML Inference</h2>
      <form onSubmit={handleSubmit} style={{ display: 'grid', gap: '0.75rem', maxWidth: '480px' }}>
        <label>
          Input vector
          <input
            type="text"
            value={vectorInput}
            onChange={(event) => setVectorInput(event.target.value)}
            placeholder="e.g. 0.5, 1.2, -0.3"
          />
        </label>
        <button type="submit" disabled={loading}>
          {loading ? 'Generating proof…' : 'Infer & generate proof'}
        </button>
        {error && <p className="error">{error}</p>}
      </form>
      <div>
        <h3>Latest proof</h3>
        <p>
          <strong>Input:</strong> {proof.input.length ? proof.input.join(', ') : 'n/a'}
        </p>
        <p>
          <strong>Prediction:</strong>{' '}
          {proof.prediction !== null ? proof.prediction.toFixed(4) : 'n/a'}
        </p>
        <p>
          <strong>Statement:</strong> {proof.statement ? <code>{proof.statement}</code> : 'n/a'}
        </p>
        <p>
          <strong>Proof:</strong>{' '}
          {proof.proof ? <code style={{ wordBreak: 'break-all' }}>{proof.proof}</code> : 'n/a'}
        </p>
        <p>
          <strong>Verified:</strong>{' '}
          <span style={{ color: proof.verified ? '#34d399' : '#f87171' }}>
            {proof.verified ? 'valid' : 'unverified'}
          </span>
        </p>
      </div>
    </section>
  );
};

export default ZkmlView;
