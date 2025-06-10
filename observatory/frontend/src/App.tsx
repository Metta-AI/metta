import { useEffect, useState } from "react";
import { ServerRepo, Repo } from "./repo";
import { Dashboard } from "./Dashboard";

function App() {
  // Data loading state
  type DefaultState = {
    type: "default";
    error: string | null;
  };
  type LoadingState = {
    type: "loading";
  };
  type RepoState = {
    type: "repo";
    repo: Repo;
  };
  type State = DefaultState | LoadingState | RepoState;

  const [state, setState] = useState<State>({ type: "loading" });

  useEffect(() => {
    const initializeRepo = async () => {
      try {
        // Get server URL from environment or use default
        const serverUrl =
          import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
        const repo = new ServerRepo(serverUrl);

        // Test the connection by calling getSuites
        await repo.getSuites();

        setState({ type: "repo", repo });
      } catch (err: any) {
        setState({
          type: "default",
          error: `Failed to connect to server: ${
            err.message
          }. Make sure the server is running at ${
            import.meta.env.VITE_SERVER_URL || "http://localhost:8000"
          }`,
        });
      }
    };

    initializeRepo();
  }, []);

  if (state.type === "default") {
    return (
      <div
        style={{
          fontFamily: "Arial, sans-serif",
          margin: 0,
          padding: "20px",
          background: "#f8f9fa",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            maxWidth: "600px",
            margin: "0 auto",
            background: "#fff",
            padding: "40px",
            borderRadius: "8px",
            boxShadow: "0 2px 4px rgba(0,0,0,.1)",
            textAlign: "center",
          }}
        >
          <h1
            style={{
              color: "#333",
              marginBottom: "20px",
            }}
          >
            Policy Evaluation Dashboard
          </h1>
          <p style={{ marginBottom: "20px", color: "#666" }}>
            Unable to connect to the evaluation server.
          </p>
          {state.error && (
            <div
              style={{ color: "red", marginTop: "10px", marginBottom: "20px" }}
            >
              {state.error}
            </div>
          )}
          <p style={{ color: "#666", fontSize: "14px" }}>
            Please ensure the server is running and accessible.
          </p>
        </div>
      </div>
    );
  }

  if (state.type === "loading") {
    return (
      <div
        style={{
          fontFamily: "Arial, sans-serif",
          margin: 0,
          padding: "20px",
          background: "#f8f9fa",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <div
          style={{
            textAlign: "center",
            color: "#666",
          }}
        >
          <h2>Connecting to server...</h2>
          <p>Loading evaluation data from the server.</p>
        </div>
      </div>
    );
  }

  if (state.type === "repo") {
    return <Dashboard repo={state.repo} />;
  }

  return null;
}

export default App;
