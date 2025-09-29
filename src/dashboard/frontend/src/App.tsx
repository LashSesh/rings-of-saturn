import React from 'react';
import LedgerView from './components/LedgerView';
import HdagView from './components/HdagView';
import SpiralView from './components/SpiralView';
import TicView from './components/TicView';
import MlView from './components/MlView';
import ZkmlView from './components/ZkmlView';

const App: React.FC = () => {
  return (
    <main>
      <header>
        <h1>Rings of Saturn Dashboard</h1>
        <p>Interactive exploration of ledger, HDAG, TIC and ZKML subsystems.</p>
      </header>
      <div className="grid">
        <LedgerView />
        <HdagView />
      </div>
      <SpiralView />
      <div className="grid">
        <TicView />
        <MlView />
      </div>
      <ZkmlView />
    </main>
  );
};

export default App;
