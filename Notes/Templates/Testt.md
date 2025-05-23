<%* 
// 1) Ask for the target folder
let folder = await tp.system.prompt("Target folder (e.g. Projects/Alpha)");

// 2) Create a “.gitkeep” in there to force the folder into existence
await tp.file.create(folder + "/.gitkeep", "");

// 3) Ask for the note title (no “.md”)
let title = await tp.system.prompt("Note title");

// 4) Build the full path
let path = folder + "/" + title + ".md";

// 5) Build frontmatter string (use simple concatenation, no backticks inside backticks)
let fm = (
    "---\n" +
    "title: \"" + title + "\"\n" +
    "date: " + tp.date.now("YYYY-MM-DD") + "\n" +
    "---\n\n"
);

// 6) Create the file with that frontmatter
await tp.file.create(path, fm);

// 7) Open it in the editor
await tp.obsidian.open_file(path);
%>
