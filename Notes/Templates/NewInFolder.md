<%* 
// 1) Ask for folder name (relative to vault root)
let folder = await tp.system.prompt("Target folder (e.g. Projects/Alpha)");

// 2) Ensure that folder exists
await tp.file.create_folder(folder);

// 3) Ask for note title (no “.md”)
let title = await tp.system.prompt("Note title");

// 4) Build path and frontmatter
let path = `${folder}/${title}.md`;
let fm = `---\ntitle: "${title}"\ndate: ${tp.date.now("YYYY-MM-DD")}\n---\n\n`;

// 5) Create the file
await tp.file.create(path, fm);

// 6) Open it
await tp.obsidian.open_file(path);
%>
