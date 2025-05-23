<%* 
// 1. Ask where to put it
let folder = await tp.system.prompt("Target folder (relative, e.g. Projects/Alpha)");
// 2. If that folder doesn’t exist yet, make it
if (!app.vault.getAbstractFileByPath(folder)) {
  await app.vault.createFolder(folder);
}

// 3. Ask for the note’s title
let title = await tp.system.prompt("Note title (no “.md”)");

// 4. Build the new path
let path = `${folder}/${title}.md`;

// 5. Create the file (with a simple frontmatter stub)
let content = `---
title: "${title}"
date: ${tp.date.now("YYYY-MM-DD")}
---
`;
await app.vault.create(path, content);

// 6. Open it in the editor
await tp.obsidian.open_file(path);
%>