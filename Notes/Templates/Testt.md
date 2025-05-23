<%*
// 1) Prompt for target folder
let folder = await tp.system.prompt("Target folder (e.g. Projects/Alpha)");

// 2) Ensure folder exists
await tp.file.create_folder(folder);

// 3) Prompt for title
let title = await tp.system.prompt("Note title");

// 4) Build path & frontmatter
let path = `${folder}/${title}.md`;
let fm = `---
title: "${title}"
date: ${tp.date.now("YYYY-MM-DD")}
---
`;

// 5) Create & open
await tp.file.create(path, fm);
await tp.obsidian.open_file(path);
%>
