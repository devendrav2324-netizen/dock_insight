import React, { useState } from 'react';
import { 
  Ship, Package, Loader2, AlertCircle, 
  TrendingUp, Activity, DollarSign, ShieldAlert, FileText, CheckCircle2, XCircle, BrainCircuit
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, 
  ResponsiveContainer, PieChart, Pie, Cell, Legend
} from 'recharts';
import { analyzeVoyage } from '../api';

const COLORS = ['#3182CE', '#63B3ED', '#38A169', '#E53E3E', '#DD6B20'];

export default function VoyagePlanner() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const futureDate = new Date();
  futureDate.setDate(futureDate.getDate() + 30);
  
  const [formData, setFormData] = useState({
    cargo_type: 'coal',
    cargo_quantity: 100000,
    origin: 'INA_TAB',
    destination: 'IND_DHA',
    required_delivery_date: futureDate.toISOString().split('T')[0],
    number_of_voyages: 3,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeVoyage(formData);
      setResult(data);
    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        if (typeof detail === 'string') {
          setError(detail);
        } else if (typeof detail === 'object' && detail.error_message) {
          setError(detail.error_message);
        } else {
          setError(JSON.stringify(detail));
        }
      } else {
        setError(err.message || "An unexpected error occurred while communicating with the intelligence engine.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      
      {/* Input Form Card */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title"><Package size={20} /> Voyage Parameters</h2>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="grid-3">
            <div className="form-group">
              <label className="form-label">Cargo Type</label>
              <select name="cargo_type" className="form-control" value={formData.cargo_type} onChange={handleChange}>
                <option value="coal">Thermal Coal</option>
                <option value="iron_ore">Iron Ore</option>
                <option value="grain">Grain</option>
                <option value="bauxite">Bauxite</option>
              </select>
            </div>
            
            <div className="form-group">
              <label className="form-label">Quantity (MT)</label>
              <input type="number" name="cargo_quantity" className="form-control" value={formData.cargo_quantity} onChange={handleChange} required />
            </div>

            <div className="form-group">
              <label className="form-label">Required Delivery</label>
              <input type="date" name="required_delivery_date" className="form-control" value={formData.required_delivery_date} onChange={handleChange} required />
            </div>
            
            <div className="form-group">
              <label className="form-label">Origin Port</label>
              <select name="origin" className="form-control" value={formData.origin} onChange={handleChange}>
                <option value="AUS_NEW">Newcastle, Australia</option>
                <option value="INA_TAB">Taboneo, Indonesia</option>
                <option value="ZAF_RB">Richards Bay, South Africa</option>
              </select>
            </div>
            
            <div className="form-group">
              <label className="form-label">Destination Port</label>
              <select name="destination" className="form-control" value={formData.destination} onChange={handleChange}>
                <option value="IND_PAR">Paradip, India</option>
                <option value="IND_VIS">Visakhapatnam, India</option>
                <option value="IND_GAN">Gangavaram, India</option>
                <option value="IND_GOP">Gopalpur, India</option>
                <option value="IND_DHA">Dhamra, India</option>
                <option value="IND_HAL">Haldia, India</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Number of Voyages</label>
              <input type="number" name="number_of_voyages" className="form-control" value={formData.number_of_voyages} onChange={handleChange} min={1} required />
            </div>
          </div>
          
          <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? <><Loader2 className="spinner" size={18} /> Analyzing...</> : <><Activity size={18} /> Analyze Voyage</>}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <div className="error-alert">
          <AlertCircle size={24} style={{ flexShrink: 0 }} />
          <div>
            <h4 style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Engine Validation Failed</h4>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Results View */}
      {result && !loading && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', animation: 'fadeIn 0.5s ease' }}>
          
          {/* Top Recommendations */}
          <div className="grid-3">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title"><Ship size={18} /> Recommended Vessel</h3>
              </div>
              <div className="stat-value accent">{result.recommended_vessel?.class || 'N/A'}</div>
              <div style={{ marginTop: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                Score: <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{result.recommended_vessel?.score ?? 0}/100</span>
              </div>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                {result.recommended_vessel?.reasons?.[0] || 'Optimal vessel selection based on multi-criteria scoring.'}
              </p>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title"><FileText size={18} /> Contract Strategy</h3>
              </div>
              <div className="stat-value">{result.contract_strategy?.recommended_strategy || 'Spot Strategy'}</div>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                {result.contract_strategy?.reasons?.[0] || result.contract_strategy?.reasoning?.[0] || 'Balanced contract strategy recommended.'}
              </p>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title"><ShieldAlert size={18} /> Overall Risk</h3>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <div className={`stat-value ${result.risk_analysis?.overall_level === 'HIGH' ? 'danger' : 'warning'}`}>
                  {result.risk_analysis?.overall_score ?? 0}/100
                </div>
                <div className="badge badge-mod">{result.risk_analysis?.overall_level || 'MODERATE'}</div>
              </div>
              <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
                Dominant factor: {result.risk_analysis?.dominant_risk || 'None'}
              </p>
            </div>
          </div>

          <div className="grid-2">
            {/* Voyage Economics */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title"><DollarSign size={18} /> Voyage Economics Breakdown</h3>
              </div>
              <div style={{ display: 'flex', height: '300px' }}>
                <div style={{ flex: 1 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Freight', value: result.voyage_economics?.breakdown?.freight_cost || 0 },
                          { name: 'Bunker', value: result.voyage_economics?.breakdown?.bunker_cost || 0 },
                          { name: 'Port Charges', value: result.voyage_economics?.breakdown?.port_charges || 0 },
                          { name: 'Demurrage', value: result.voyage_economics?.breakdown?.demurrage_risk || 0 },
                        ]}
                        cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5}
                        dataKey="value" stroke="none"
                      >
                        {[0, 1, 2, 3].map((_, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip formatter={(value) => `$${Number(value).toLocaleString()}`} contentStyle={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: '8px' }}/>
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div className="stat-box" style={{ marginBottom: '1rem' }}>
                    <div className="stat-label">Total Est. Cost</div>
                    <div className="stat-value">${(result.voyage_economics?.total_cost || 0).toLocaleString()}</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">Cost per Tonne</div>
                    <div className="stat-value">${(result.voyage_economics?.cost_per_tonne || 0).toFixed(2)}</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Freight Forecast */}
            <div className="card">
              <div className="card-header">
                <h3 className="card-title"><TrendingUp size={18} /> Freight Rate Forecast</h3>
              </div>
              <div style={{ height: '300px' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={[
                    { day: 'T-3', rate: (result.market_forecast?.current_rate ?? result.market_forecast?.current_rate_usd ?? 15) * 0.96 },
                    { day: 'T-2', rate: (result.market_forecast?.current_rate ?? result.market_forecast?.current_rate_usd ?? 15) * 0.98 },
                    { day: 'T-1', rate: (result.market_forecast?.current_rate ?? result.market_forecast?.current_rate_usd ?? 15) * 0.99 },
                    { day: 'T0', rate: result.market_forecast?.current_rate ?? result.market_forecast?.current_rate_usd ?? 15.0 },
                    { day: 'T+1', rate: (result.market_forecast?.predicted_rate ?? result.market_forecast?.forecast_rate_usd ?? 15.5) * 0.98 },
                    { day: 'T+2', rate: (result.market_forecast?.predicted_rate ?? result.market_forecast?.forecast_rate_usd ?? 15.5) * 0.99 },
                    { day: 'T+3', rate: result.market_forecast?.predicted_rate ?? result.market_forecast?.forecast_rate_usd ?? 15.5 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" vertical={false} />
                    <XAxis dataKey="day" stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} domain={['auto', 'auto']} tickFormatter={(value) => `$${Number(value).toFixed(1)}`} />
                    <RechartsTooltip contentStyle={{ backgroundColor: 'var(--bg-tertiary)', border: '1px solid var(--border-color)', borderRadius: '8px' }} />
                    <Line type="monotone" dataKey="rate" stroke="var(--accent-primary)" strokeWidth={3} dot={{ r: 4, fill: 'var(--accent-primary)', strokeWidth: 0 }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Decision Explanation Center */}
          {result.explanation && (
            <div className="card" style={{ borderLeft: '4px solid var(--accent-primary)', backgroundColor: 'rgba(49, 130, 206, 0.02)' }}>
              <div className="card-header" style={{ marginBottom: '1.5rem' }}>
                <h3 className="card-title" style={{ color: 'var(--accent-primary)' }}><BrainCircuit size={20} /> Decision Explanation Center</h3>
              </div>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div>
                  <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Primary Decision Factors</h4>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {result.explanation.primary_reasons?.map((reason: string, idx: number) => (
                      <li key={idx} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start', fontSize: '0.95rem' }}>
                        <CheckCircle2 size={18} color="#10B981" style={{ flexShrink: 0, marginTop: '2px' }} />
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                
                {result.explanation.alternatives_rejected && result.explanation.alternatives_rejected.length > 0 && (
                  <div style={{ marginTop: '0.5rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-light)' }}>
                    <h4 style={{ fontSize: '0.95rem', color: 'var(--text-secondary)', marginBottom: '1rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Rejected Alternatives</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1rem' }}>
                      {result.explanation.alternatives_rejected.map((alt: any, idx: number) => (
                        <div key={idx} style={{ backgroundColor: 'white', padding: '1rem', borderRadius: '0.5rem', border: '1px solid var(--border-light)' }}>
                          <div style={{ fontWeight: 600, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <XCircle size={16} color="#EF4444" />
                            {alt.vessel_class}
                          </div>
                          <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                            {alt.reasons_rejected?.map((r: string, ridx: number) => (
                              <li key={ridx}>• {r}</li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
          
        </div>
      )}
    </div>
  );
}

