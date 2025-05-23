<%* 
// capture the source (the “Untitled” file you started in)
const src = tp.file.path();

// 1) Ask for the target folder (relative to vault root)
let folder = await tp.system.prompt("Target folder (e.g. Test)");

// 2) If it doesn’t exist, create it
if (!app.vault.getAbstractFileByPath(folder)) {
  await app.vault.createFolder(folder);
}

// 3) Ask for the note title (no “.md”)
let title = await tp.system.prompt("Note title");

// 4) Build the full path
let dest = `${folder}/${title}.md`;

// 5) Build a simple frontmatter + blank body
let content = 
  "---\n" +
  `title: "${title}"\n` +
  `date: ${tp.date.now("YYYY-MM-DD")}\n` +
  "---\n\n";

// 6) Create the new file
await app.vault.create(dest, content);

// 7) Open it in the editor
await app.workspace.openLinkText(dest, '', false);

// 8) Delete the original “Untitled” file
const orig = app.vault.getAbstractFileByPath(src);
if (orig) {
  await app.vault.delete(orig);
}
%>
