---

---
```
title: "Log Entire - <% tp.date.now("YYYY__MM__DD") %>"
created: "<% tp.date.now("YYYY__MM__DD") %>"
last_modified: "<%+ tp.file.last_modified_date("YYYY-MM-DD") %>"
tags: [research-log]
 <% await tp.file.move("Research Log/"+"Log Entire -"+tp.date.now("YYYY__MM__DD"))   %>
```



# Log Entire - <% tp.date.now("YYYY__MM__DD") %>  
_Last modified: <%+ tp.file.last_modified_date("YYYY-MM-DD") %>_

## Tasks and Goals
- [ ] <% tp.file.cursor() %>

## Findings and Notes
- 

## Actions Taken
1. 

## Future Plans
- 
