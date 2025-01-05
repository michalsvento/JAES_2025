document.addEventListener("DOMContentLoaded", () => {
  let selectedRoot = "";

  // Set the root folder
  function setRoot(root) {
    selectedRoot = root;
    const rootDisplay = document.getElementById("selected-root-display");
    if (rootDisplay) rootDisplay.textContent = `Selected Root: ${selectedRoot}`;
    updatePlayButtonPaths();
  }

  // Update play button paths with the selected root
  function updatePlayButtonPaths() {
    document.querySelectorAll(".play-audio").forEach((button) => {
      const file = button.getAttribute("data-file");
      button.setAttribute("data-path", selectedRoot + file);
    });
  }

  // Play audio
  function playAudio(button) {
    const audioPath = button.getAttribute("data-path");
    if (!audioPath) {
      alert("Please select a root folder first!");
      return;
    }
    const audioPlayer = document.getElementById("shared-audio-player");
    if (audioPlayer) {
      audioPlayer.src = audioPath;
      audioPlayer.play();
      console.log(`Playing: ${audioPath}`);
    }
  }

  // Assign event handlers
  document.querySelectorAll(".select-root").forEach((button) => {
    button.addEventListener("click", () => setRoot(button.getAttribute("data-root")));
  });

  document.querySelectorAll(".play-audio").forEach((button) => {
    button.addEventListener("click", () => playAudio(button));
  });
});