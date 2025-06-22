import { useEffect, useState } from "react";
import { ServerRepo, Repo } from "./repo";
import { Dashboard } from "./Dashboard";
import { TokenManager } from "./TokenManager";
import { config } from "./config";

// CSS for navigation
const NAV_CSS = `
.nav-container {
  background: #fff;
  border-bottom: 1px solid #ddd;
  padding: 0 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.1);
}

.nav-content {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.nav-brand {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.nav-tabs {
  display: flex;
  gap: 0;
}

.nav-tab {
  padding: 15px 20px;
  text-decoration: none;
  color: #666;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.nav-tab:hover {
  color: #333;
  background: #f8f9fa;
}

.nav-tab.active {
  color: #007bff;
  border-bottom-color: #007bff;
}

.page-container {
  padding-top: 0;
}
`;

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
  const [currentPage, setCurrentPage] = useState<"dashboard" | "tokens">("dashboard");

  useEffect(() => {
    const initializeRepo = async () => {
      const serverUrl = config.apiBaseUrl;
      try {
        const repo = new ServerRepo(serverUrl);

        // Test the connection by calling getSuites
        await repo.getSuites();

        setState({ type: "repo", repo });
      } catch (err: any) {
        setState({
          type: "default",
          error: `Failed to connect to server: ${
            err.message
          }. Make sure the server is running at ${serverUrl}`,
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
    return (
      <div style={{ fontFamily: "Arial, sans-serif", margin: 0 }}>
        <style>{NAV_CSS}</style>
        <nav className="nav-container">
          <div className="nav-content">
            <a href="#" className="nav-brand" onClick={(e) => { e.preventDefault(); setCurrentPage("dashboard"); }}>
              Policy Evaluation Dashboard
            </a>
            <div className="nav-tabs">
              <a
                href="#"
                className={`nav-tab ${currentPage === "dashboard" ? "active" : ""}`}
                onClick={(e) => { e.preventDefault(); setCurrentPage("dashboard"); }}
              >
                Dashboard
              </a>
              <a
                href="#"
                className={`nav-tab ${currentPage === "tokens" ? "active" : ""}`}
                onClick={(e) => { e.preventDefault(); setCurrentPage("tokens"); }}
              >
                Token Management
              </a>
            </div>
          </div>
        </nav>

        <div className="page-container">
          {currentPage === "dashboard" ? (
            <Dashboard repo={state.repo} />
          ) : (
            <TokenManager repo={state.repo} />
          )}
        </div>
      </div>
    );
  }

  return null;
}

export default App;
