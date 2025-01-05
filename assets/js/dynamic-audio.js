document.addEventListener("DOMContentLoaded", function () {
  const rootButtons = document.querySelectorAll(".select-root");
  const audioButtons = document.querySelectorAll("#audioTable button");

  let selectedRoot = "";

  // Handle clicks on the root selection buttons
  rootButtons.forEach((button) => {
    button.addEventListener("click", function () {
      selectedRoot = this.getAttribute("data-root");

      // Update the second table's audio paths dynamically
      audioButtons.forEach((btn, index) => {
        const audioFile = `clean_${index}.wav`;
        btn.setAttribute("onclick", `playAudio('${selectedRoot}${audioFile}')`);
      });
    });
  });

  function playAudio(audioPath) {
    const audioPlayer = document.getElementById("hard-clipping-player");
    audioPlayer.src = audioPath;
    audioPlayer.play();
  }
});