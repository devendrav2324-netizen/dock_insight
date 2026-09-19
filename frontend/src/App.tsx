import { useState, useEffect } from 'react';
import { 
  Ship, Anchor, TrendingUp, AlertTriangle, FileText, 
  Settings, LayoutDashboard, Search, ShipWheel
} from 'lucide-react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import VoyagePlanner from './pages/VoyagePlanner';

// Layout Component
const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  const [demoMode, setDemoMode] = useState(false);

  useEffect(() => {
    import('./api').then(({ getHealth }) => {
      getHealth().then(data => {
        if (data.sih_demo_mode) {
          setDemoMode(true);
        }
      }).catch(err => console.error("Health check failed", err));
    });
  }, []);

  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Voyage Planner', path: '/voyage-planner', icon: ShipWheel },
    { name: 'Freight Forecast', path: '/forecast', icon: TrendingUp },
    { name: 'Vessel Optimizer', path: '/optimizer', icon: Ship },
    { name: 'Port Intelligence', path: '/ports', icon: Anchor },
    { name: 'Risk Center', path: '/risk', icon: AlertTriangle },
    { name: 'Contract Strategy', path: '/contracts', icon: FileText },
  ];

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <Ship className="logo-icon" size={24} />
            CharterAI
          </div>
        </div>
        <nav className="sidebar-nav">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;
            return (
              <Link 
                key={item.name} 
                to={item.path} 
                className={`nav-item ${isActive ? 'active' : ''}`}
              >
                <Icon size={18} />
                {item.name}
              </Link>
            );
          })}
        </nav>
      </aside>

      <main className="main-content">
        <header className="top-header">
          <div className="page-title">
            {navItems.find(item => item.path === location.pathname)?.name || 'CharterAI'}
          </div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center', color: 'var(--text-secondary)' }}>
            <Search size={20} />
            <Settings size={20} />
            <div style={{ width: 32, height: 32, borderRadius: '50%', backgroundColor: 'var(--accent-primary)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>
              JD
            </div>
          </div>
        </header>
        
        {demoMode && (
          <div style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444', padding: '0.75rem 1.5rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.5rem', borderBottom: '1px solid rgba(239, 68, 68, 0.2)' }}>
            <AlertTriangle size={18} />
            SIH Demo Mode Active: The current environment is using synthetic [DEMO/SYNTHETIC] database values for vessels, ports, and economics to avoid external API dependencies.
          </div>
        )}
        
        <div className="content-area">
          {children}
        </div>
      </main>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<div><h2>Dashboard Overview</h2><p style={{marginTop: '1rem', color: 'var(--text-secondary)'}}>Welcome to CharterAI. Select Voyage Planner to begin.</p></div>} />
          <Route path="/voyage-planner" element={<VoyagePlanner />} />
          <Route path="/forecast" element={<div>Forecast Module</div>} />
          <Route path="/optimizer" element={<div>Optimizer Module</div>} />
          <Route path="/ports" element={<div>Ports Module</div>} />
          <Route path="/risk" element={<div>Risk Module</div>} />
          <Route path="/contracts" element={<div>Contracts Module</div>} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
