import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

type HdagNode = {
  id: string;
  vector: number[];
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

type HdagEdge = {
  source: string;
  target: string;
  weight: number;
};

type HdagResponse = {
  nodes: HdagNode[];
  edges: HdagEdge[];
};

const useHdagData = () => {
  const [data, setData] = useState<HdagResponse>({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await fetch('/dashboard/hdag');
        if (!response.ok) {
          throw new Error(`Failed to load HDAG: ${response.statusText}`);
        }
        const payload = (await response.json()) as HdagResponse;
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

  return { data, loading, error };
};

const HdagView: React.FC = () => {
  const { data, loading, error } = useHdagData();
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) {
      return;
    }
    const width = svgElement.clientWidth || 600;
    const height = 360;

    const svg = d3.select(svgElement);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${width} ${height}`);

    if (data.nodes.length === 0) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#9ca3af')
        .text('No nodes available. Use the API to seed the HDAG.');
      return;
    }

    const simulation = d3
      .forceSimulation(data.nodes)
      .force(
        'link',
        d3
          .forceLink(data.edges)
          .id((d: d3.SimulationNodeDatum & HdagNode) => d.id)
          .distance((edge) => {
            const weight = (edge as unknown as HdagEdge).weight ?? 0.5;
            return 180 - 60 * weight;
          })
      )
      .force('charge', d3.forceManyBody().strength(-250))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const link = svg
      .append('g')
      .attr('stroke', 'rgba(255,255,255,0.4)')
      .selectAll('line')
      .data(data.edges)
      .enter()
      .append('line')
      .attr('stroke-width', (d) => Math.max(1, d.weight * 2));

    const node = svg
      .append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', 14)
      .attr('fill', '#6a5acd')
      .call(
        d3
          .drag<SVGCircleElement, HdagNode>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    const labels = svg
      .append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .text((d) => d.id)
      .attr('fill', '#f4f6fb')
      .attr('font-size', 12)
      .attr('text-anchor', 'middle')
      .attr('dy', 28);

    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => (d.source.x as number) ?? 0)
        .attr('y1', (d: any) => (d.source.y as number) ?? 0)
        .attr('x2', (d: any) => (d.target.x as number) ?? 0)
        .attr('y2', (d: any) => (d.target.y as number) ?? 0);

      node.attr('cx', (d: any) => d.x as number).attr('cy', (d: any) => d.y as number);
      labels
        .attr('x', (d: any) => d.x as number)
        .attr('y', (d: any) => (d.y as number) ?? 0);
    });

    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <section aria-labelledby="hdag-heading">
      <h2 id="hdag-heading">HDAG</h2>
      {loading && <p>Loading HDAG…</p>}
      {error && <p className="error">{error}</p>}
      <svg ref={svgRef} role="img" aria-label="HDAG force-directed graph" />
    </section>
  );
};

export default HdagView;
