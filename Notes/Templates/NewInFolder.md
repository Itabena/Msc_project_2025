<%* 
// 1) Prompt for the folder and force-create it
let folder = await tp.system.prompt("Target folder (e.g. Test)");
await tp.file.create(`${folder}/.gitkeep`, "");

// 2) Prompt for the final note title
let title = await tp.system.prompt("Note title");
title = title.replace(/[\/\\]/g, '-').trim();

// 3) Build paths
const src = tp.file.path();            // e.g. "Untitled.md"
const dest = `${folder}/${title}.md`;  // e.g. "Test/My Note.md"

// 4) Rename (move) the current file into its final home
await tp.file.move(src, dest);

// 5) Prepare frontmatter + body
const fm = [
  '---',
  `title: "${title}"`,
  `date: ${tp.date.now("YYYY-MM-DD")}`,
  '---',
  ''
].join('\n');

// 6) Overwrite the moved file’s contents
const file = app.vault.getAbstractFileByPath(dest);
await app.vault.modify(file, fm);

// 7) Open it for you to continue writing
await app.workspace.openLinkText(dest, '', false);
%>
