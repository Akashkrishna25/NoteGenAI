import { useEffect, useState } from "react";
import "./App.css";

function App() {
  // =========================
  // STATES
  // =========================

  const [topic, setTopic] = useState("");
  const [level, setLevel] = useState("beginner");
  const [length, setLength] = useState("medium");

  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Dark mode
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem("notegen-theme") === "dark";
  });

  // =========================
  // SAVE THEME
  // =========================

  useEffect(() => {
    localStorage.setItem(
      "notegen-theme",
      darkMode ? "dark" : "light"
    );
  }, [darkMode]);

  // =========================
  // EXAMPLE TOPICS
  // =========================

  const examples = [
    "Artificial Intelligence",
    "Machine Learning",
    "Operating System",
    "Transformers",
    "Computer Networks",
  ];

  // =========================
  // GENERATE NOTES
  // =========================

  const generateNotes = async () => {
    if (!topic.trim()) {
      setError("Please enter a topic.");
      return;
    }

    setLoading(true);
    setError("");
    setNotes("");

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/generate-notes",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            topic: topic.trim(),
            level: level,
            length: length,
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to generate notes.");
      }

      const data = await response.json();

      setNotes(data.notes);
    } catch (err) {
      console.error(err);

      setError(
        "Could not connect to NoteGen AI. Make sure the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // =========================
  // COPY NOTES
  // =========================

  const copyNotes = async () => {
    if (!notes) return;

    try {
      await navigator.clipboard.writeText(notes);

      alert("Notes copied successfully!");
    } catch (error) {
      alert("Unable to copy notes.");
    }
  };

  // =========================
  // DOWNLOAD NOTES
  // =========================

  const downloadNotes = () => {
    if (!notes) return;

    const content = `${topic}\n\n${notes}`;

    const blob = new Blob([content], {
      type: "text/plain",
    });

    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");

    link.href = url;

    link.download = `${topic.replace(
      /\s+/g,
      "_"
    )}_notes.txt`;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  // =========================
  // CLEAR
  // =========================

  const clearNotes = () => {
    setTopic("");
    setNotes("");
    setError("");
  };

  // =========================
  // ENTER KEY
  // =========================

  const handleKeyDown = (event) => {
    if (
      event.key === "Enter" &&
      (event.ctrlKey || event.metaKey)
    ) {
      generateNotes();
    }
  };

  // =========================
  // WORD COUNT
  // =========================

  const wordCount = notes
    ? notes.split(/\s+/).filter(Boolean).length
    : 0;

  // =========================
  // UI
  // =========================

  return (
    <div className={`app ${darkMode ? "dark" : "light"}`}>

      {/* =========================
          NAVBAR
      ========================= */}

      <nav className="navbar">

        <div className="logo">
          📚
          <span>NoteGen AI</span>
        </div>

        <div className="nav-right">

          <div className="nav-status">
            <span className="status-dot"></span>
            AI Ready
          </div>

          <button
            className="theme-toggle"
            onClick={() => setDarkMode(!darkMode)}
          >
            {darkMode ? "☀️ Light" : "🌙 Dark"}
          </button>

        </div>

      </nav>

      {/* =========================
          MAIN
      ========================= */}

      <main>

        {/* HERO */}

        <section className="hero">

          <div className="badge">
            ✨ AI Study Assistant
          </div>

          <h1>
            Learn Smarter.
            <br />

            <span>Generate Better Notes.</span>
          </h1>

          <p>
            Enter any topic and let your AI generate
            clear, structured study notes.
          </p>

        </section>

        {/* =========================
            GENERATOR CARD
        ========================= */}

        <section className="generator-card">

          <div className="section-title">

            <h2>
              Generate Notes
            </h2>

            <p>
              Tell the AI what you want to learn.
            </p>

          </div>

          {/* TOPIC */}

          <div className="input-group">

            <label>
              Topic
            </label>

            <textarea
              value={topic}
              onChange={(e) =>
                setTopic(e.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="What do you want to learn?"
              rows="3"
            />

          </div>

          {/* EXAMPLES */}

          <div className="examples">

            <span>
              Try:
            </span>

            {examples.map((example) => (
              <button
                key={example}
                className="example-btn"
                onClick={() =>
                  setTopic(example)
                }
              >
                {example}
              </button>
            ))}

          </div>

          {/* OPTIONS */}

          <div className="options">

            {/* LEVEL */}

            <div className="input-group">

              <label>
                Difficulty
              </label>

              <select
                value={level}
                onChange={(e) =>
                  setLevel(e.target.value)
                }
              >

                <option value="beginner">
                  🌱 Beginner
                </option>

                <option value="intermediate">
                  📘 Intermediate
                </option>

                <option value="advanced">
                  🚀 Advanced
                </option>

              </select>

            </div>

            {/* LENGTH */}

            <div className="input-group">

              <label>
                Note Length
              </label>

              <select
                value={length}
                onChange={(e) =>
                  setLength(e.target.value)
                }
              >

                <option value="short">
                  ⚡ Short
                </option>

                <option value="medium">
                  📖 Medium
                </option>

                <option value="long">
                  📚 Long
                </option>

              </select>

            </div>

          </div>

          {/* ERROR */}

          {error && (
            <div className="error">
              ⚠️ {error}
            </div>
          )}

          {/* GENERATE BUTTON */}

          <button
            className="generate-btn"
            onClick={generateNotes}
            disabled={loading}
          >

            {loading ? (
              <>
                <span className="spinner"></span>
                Generating Notes...
              </>
            ) : (
              <>
                ✨ Generate Notes
              </>
            )}

          </button>

          <div className="shortcut">
            Tip: Press <b>Ctrl + Enter</b> to generate
          </div>

        </section>

        {/* =========================
            GENERATED NOTES
        ========================= */}

        {notes && (

          <section className="notes-card">

            {/* HEADER */}

            <div className="notes-header">

              <div>

                <span className="notes-label">
                  GENERATED NOTES
                </span>

                <h2>
                  {topic}
                </h2>

                <p>
                  {level} • {length}
                </p>

              </div>

              {/* ACTIONS */}

              <div className="notes-actions">

                <button
                  onClick={copyNotes}
                >
                  📋 Copy
                </button>

                <button
                  onClick={downloadNotes}
                >
                  ⬇ Download
                </button>

                <button
                  onClick={clearNotes}
                >
                  ✕ Clear
                </button>

              </div>

            </div>

            {/* NOTES */}

            <div className="notes-content">
              {notes}
            </div>

            {/* FOOTER */}

            <div className="notes-footer">

              <span>
                🧠 Generated by your custom Transformer LLM
              </span>

              <span>
                {wordCount} words
              </span>

            </div>

          </section>

        )}

      </main>

      {/* =========================
          FOOTER
      ========================= */}

      <footer>

        <p>
          NoteGen AI • Built with a Transformer LLM
        </p>

      </footer>

    </div>
  );
}

export default App;