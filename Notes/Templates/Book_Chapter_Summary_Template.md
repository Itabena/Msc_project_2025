---
<%* 
// Prompt for book chapter metadata
let bookTitle     = await tp.system.prompt("Book title (With sapces and cap 1st letter)");
let chapterNumber = await tp.system.prompt("Chapter number");
let bookAuthor    = await tp.system.prompt("Book author");
let pubYear       = await tp.system.prompt("Publication year");
let pages         = await tp.system.prompt("Pages (e.g. 12–34)");
let Subject         = await tp.system.prompt("Subject (like astphys)");
 await tp.file.move("Articles and books/"+"Books_"+Subject+"/"+BookTitle+"Chapters"+chapterNumber)   
%>

title: "Book: <% bookTitle %> Ch. <% chapterNumber %>"
book: "<% bookTitle %>"
chapter: <% chapterNumber %>
author: "<% bookAuthor %>"
year: <% pubYear %>
pages: "<% pages %>"
tags: [book-summary]
---

# `<% bookTitle %>` — Chapter <% chapterNumber %>

**Author:** <% bookAuthor %>  
**Published:** <% pubYear %>  
**Pages:** <% pages %>
**File link:** [Zotero Link](zotero://open-pdf/library/items/<% doi %>) 

---

## 1. Chapter Overview
_A 2–3 sentence high-level description._

## 2. Major Themes & Concepts
- 

## 3. Important Terms & Definitions
| Term        | Definition                               |
|-------------|------------------------------------------|
|             |                                          |

## 4. Key Arguments or Narratives
1.  
2.  

## 5. Memorable Quotes
>  

## 6. Reflections & Takeaways
- 
