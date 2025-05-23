---

---

<%* 
// 1) capture today’s date in your desired format
const today  = tp.date.now("YYYY__MM__DD");

// 2) get the file’s last‐modified date
const lastMod = tp.file.last_modified_date("YYYY-MM-DD");
%>
title: "Log Entire - <% today %>"
created: "<% today %>"
last_modified: "<% lastMod %>"
tags: [research-log]
---

# Log Entire - <% today %>  
_Last modified: <% lastMod %>_

## Tasks and Goals
- [ ] <% tp.file.cursor() %>

## Findings and Notes
- 

## Actions Taken
1. 

## Future Plans
- 
