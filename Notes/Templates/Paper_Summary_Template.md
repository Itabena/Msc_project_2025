---
<%* 
// Prompt for metadata
let citekey     = await tp.system.prompt("Citation key (e.g. smith2020)");
let paperTitle  = await tp.system.prompt("Paper title");
let authors     = await tp.system.prompt("Authors");
let year        = await tp.system.prompt("Publication year");
let journal     = await tp.system.prompt("Journal name");
let doi         = await tp.system.prompt("(folder in library)/(File name) from zotero library.");
%>

title: "<% citekey %> Paper Summary"
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
**File link:** [Zotero Link](zotero://open-pdf/library/items/<% doi %>) 
**Last Modified:**  <% tp.file.last_modified_date("YYYY-MM-DD") %>

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
