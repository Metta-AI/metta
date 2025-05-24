import { useEffect, useState } from 'react'
import { loadDataFromUri, loadDataFromFile } from './data_loader'
import { DataRepo, Repo } from './repo'
import { Dashboard } from './Dashboard'

function App() {
  // Data loading state
  const [dataUri, setDataUri] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fileError, setFileError] = useState<string | null>(null)
  const [repo, setRepo] = useState<Repo | null>(null)

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const dataUri = params.get('data');
    if (dataUri) {
      setDataUri(dataUri);
    }
  }, []);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        if (dataUri) {
          const data = await loadDataFromUri(dataUri);
          setRepo(new DataRepo(data));
        }
        setLoading(false);
      } catch (err: any) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadData();
  }, [dataUri]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setLoading(true);
      setError(null);
      setFileError(null);

      const data = await loadDataFromFile(file);
      setRepo(new DataRepo(data));
      setLoading(false);
    } catch (err: any) {
      setFileError('Invalid file format. Please upload a valid JSON file.');
      setLoading(false);
    }
  };

  if (!dataUri && !repo) {
    return (
      <div style={{ 
        fontFamily: 'Arial, sans-serif',
        margin: 0,
        padding: '20px',
        background: '#f8f9fa',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{
          maxWidth: '600px',
          margin: '0 auto',
          background: '#fff',
          padding: '40px',
          borderRadius: '8px',
          boxShadow: '0 2px 4px rgba(0,0,0,.1)',
          textAlign: 'center'
        }}>
          <h1 style={{
            color: '#333',
            marginBottom: '20px'
          }}>
            Policy Evaluation Dashboard
          </h1>
          <p style={{ marginBottom: '20px', color: '#666' }}>
            Upload your evaluation data or provide a data URI as a query parameter.
          </p>
          <div style={{ marginBottom: '20px' }}>
            <input
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              style={{
                padding: '10px',
                border: '2px dashed #ddd',
                borderRadius: '4px',
                width: '100%',
                cursor: 'pointer'
              }}
            />
          </div>
          {fileError && (
            <div style={{ color: 'red', marginTop: '10px' }}>
              {fileError}
            </div>
          )}
          <p style={{ color: '#666', fontSize: '14px' }}>
            Or add <code>?data=YOUR_DATA_URI</code> to the URL
          </p>
        </div>
      </div>
    );
  }

  if (loading) {
    return <div>Loading data...</div>
  }
  
  if (error) {
    return <div style={{ color: 'red' }}>Error: {error}</div>
  }

  if (!repo) {
    return null;
  }

  return <Dashboard repo={repo} />
}

export default App 