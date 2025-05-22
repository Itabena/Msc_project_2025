---
<%* 

// 1) Compute dates
const creationDate = tp.date.now("YYYY__MM__DD");
// 2) For a first draft, lastMod = creation; you can later update via a save-trigger
var lastModDate = tp.date.now("YYYY__MM__DD");

// 3) Build the filename for convenience
const filename = `Log - ${creationDate}`;

// 4) Move the new file into a “ResearchLogs” 
const srcPath = tp.file.path(relative: boolian = True);
const destPath = `Notes\Research Log/${filename}.md`;

//await tp.file.move(srcPath,destPath)

// 3. Open the moved file in the editor
//await tp.obsidian.open_file(destPath);
%>


title: "<% filename %>"
created: "<% creationDate %>"
last_modified: "<% lastModDate %>"
tags: [research-log]
---


---
# <% filename %>  
_Last modified: <% lastModDate %>_

<% srcPath%>
<%destPath%>
## Tasks and Goals
- [ ] <% tp.file.cursor() %>

## Findings and Notes
- 

## Actions Taken
1. 

## Future Plans
- 
