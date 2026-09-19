import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface AnalyzeVoyagePayload {
  cargo_type: string;
  cargo_quantity: number;
  origin: string;
  destination: string;
  required_delivery_date: string;
  number_of_voyages: number;
}

export const analyzeVoyage = async (payload: AnalyzeVoyagePayload) => {
  const response = await apiClient.post('/analyze-voyage', payload);
  return response.data;
};

export const getHealth = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};
