import axios from 'axios';

const API_BASE_URL = '/api';

interface RagaPrediction {
  raga: string;
  probability: number;
  info: {
    aroha?: string[];
    avaroha?: string[];
    time?: string;
    mood?: string;
  };
}

export interface RagaDetectionResult {
  predictions: RagaPrediction[];
  metadata: {
    model_version: string;
    inference_time: number;
    timestamp: string;
  };
}

export async function detectRaga(
  file: FormData,
  onUploadProgress?: (progressEvent: ProgressEvent) => void
): Promise<RagaDetectionResult> {
  const response = await axios.post<RagaDetectionResult>(
    `${API_BASE_URL}/ragas/detect`,
    file,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress,
    }
  );
  return response.data;
}

export async function getSupportedRagas(): Promise<string[]> {
  const response = await axios.get<{ ragas: string[] }>(
    `${API_BASE_URL}/ragas/supported-ragas`
  );
  return response.data.ragas;
}

export async function getRagaDetails(ragaName: string): Promise<any> {
  const response = await axios.get(`${API_BASE_URL}/ragas/${encodeURIComponent(ragaName)}`);
  return response.data;
}
