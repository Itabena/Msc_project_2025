---
<%* 
// Prompt for derivation name
let name = await tp.system.prompt("Derivation name");
let today = tp.date.now("YYYY-MM-DD");
%>

title: "Derivation: <% name %>"
date: "<% today %>"
tags: [derivation, math]
---

# Derivation: <% name %>

## 1. Objective
_What are we deriving and why?_

## 2. Starting Equations / Assumptions
\[
<% tp.file.cursor() %>
\]

## 3. Step-by-Step Derivation
1. **Step 1:** _Describe transformation_  
   \[
     % YOUR MATH HERE %
   \]
2. **Step 2:** _Next manipulation_  
   \[
     % … %
   \]
3. **Final Result:**  
   \[
     \boxed{\,\text{Result here}\,}
   \]

## 4. Discussion of Approximations
- 

## 5. Special Cases & Consistency Checks
- Case \(r \to \infty\): …  
- Dimensional check: …  

## 6. References
- <% await tp.system.prompt("Enter citekeys separated by semicolons") %>
