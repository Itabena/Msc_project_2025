
```
---
creation date: <% tp.file.creation_date() %>
modification date: <% tp.file.last_modified_date("dddd Do MMMM YYYY HH:mm:ss") %>
---

<< [[<% tp.date.now("YYYY-MM-DD", -1) %>]] | [[<% tp.date.now("YYYY-MM-DD", 1) %>]] >>

# <% tp.file.title %>

<% tp.web.daily_quote() %>
```
<%* 
// Prompt for metadata up front
let citekey = await tp.system.prompt("Citation key (e.g. smith2020)");
let paperTitle = await tp.system.prompt("Paper title");
let authors = await tp.system.prompt("Authors");
let year = await tp.system.prompt("Year");
let journal = await tp.system.prompt("Journal");
let doi = await tp.system.prompt("DOI or URL");
let fileTitle = `Paper Summary: ${citekey}`;
%>
---
title: "<% fileTitle %>"
citekey: "<% citekey %>"
paper: "<% paperTitle %>"
authors: "<% authors %>"
year: <% year %>
journal: "<% journal %>"
doi: "<% doi %>"
tags: [paper-summary]
---

# <% paperTitle %> (<% year %>)  
**Authors:** <% authors %>  
**Journal:** <% journal %>  
**DOI/URL:** <% doi %>  

---

## 1. Abstract in Brief  
> 

## 2. Key Contributions  
- 

## 3. Methods & Approach  
- 

## 4. Results & Conclusions  
- 

## 5. Strengths & Weaknesses  
- **Strengths:**  
  -  
- **Weaknesses / Open Questions:**  
  -  

## 6. Relation to My Work  
- 
