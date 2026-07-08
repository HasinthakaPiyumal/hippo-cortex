# University of Kelaniya BSc Honours Thesis Template

A comprehensive LaTeX template for BSc Honours theses at the **University of Kelaniya**, Faculty of Science, Software Engineering Teaching Unit.

---

## 📁 Template Structure

```
thesis-template/
├── thesis.tex              ← Main file (compile this)
├── uokthesis.cls           ← Document class (don't edit)
├── latexmkrc               ← Auto-compilation config
├── README.md               ← This file
│
├── images/
│   └── uok_logo.png        ← University logo
│
├── frontmatter/
│   ├── acknowledgement.tex ← Your acknowledgements
│   ├── abstract.tex        ← Your abstract
│   └── abbreviations.tex   ← List of abbreviations
│
├── chapters/
│   ├── chapter01/          ← Introduction
│   │   ├── chapter01.tex
│   │   └── figures/        ← Chapter 1 figures
│   ├── chapter02/          ← Literature Review
│   ├── chapter03/          ← Methodology
│   ├── chapter04/          ← Results & Analysis
│   └── chapter05/          ← Conclusion
│
├── appendices/
│   ├── appendixA.tex       ← Source Code
│   └── appendixB.tex       ← Supplementary Materials
│
└── references/
    └── references.bib      ← Your bibliography
```

---

## 🚀 Quick Start

### Option 1: Overleaf (Recommended for Beginners)

1. Download this template as a **ZIP file**
2. Go to [Overleaf](https://www.overleaf.com) and sign up/log in
3. Click **New Project** → **Upload Project**
4. Upload the ZIP file
5. Click **Recompile** to see your thesis!

### Option 2: Local Installation

**Requirements:** TeX Live (Linux/Mac) or MiKTeX (Windows)

```bash
# Compile (run these commands in order)
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex

# OR use latexmk (automatic)
latexmk -pdf thesis.tex

# Clean auxiliary files
latexmk -c
```

---

## ✏️ Step-by-Step Guide

### Step 1: Update Your Information

Open `thesis.tex` and update these lines:

```latex
\thesistitle{Your Research Title}
\studentname{Your Full Name}
\studentno{SE/20XX/XXX}
\academicyear{20XX/20XX}
\supervisorname{Dr. Supervisor Name}
\submissiondate{Month Year}
```

### Step 2: Write Your Chapters

Edit the files in `chapters/` folder:

| File | Content |
|------|---------|
| `chapter01/chapter01.tex` | Introduction, objectives, scope |
| `chapter02/chapter02.tex` | Literature review, related work |
| `chapter03/chapter03.tex` | Methodology, implementation |
| `chapter04/chapter04.tex` | Results, analysis, discussion |
| `chapter05/chapter05.tex` | Conclusion, future work |

### Step 3: Add References

Add your sources to `references/references.bib`:

```bibtex
@article{smith2023,
    author  = {Smith, John},
    title   = {Article Title},
    journal = {Journal Name},
    year    = {2023},
    volume  = {10},
    pages   = {1--15}
}
```

Cite in your text:
```latex
According to Smith \cite{smith2023}, this is important.
Several studies \cite{smith2023, jones2022} have shown...
```

### Step 4: Add Figures

1. Place image files in `chapters/chapterXX/figures/`
2. Include in your chapter:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{your_image}
    \caption{Description of your figure}
    \label{fig:unique_label}
\end{figure}
```

Reference with: `As shown in Figure~\ref{fig:unique_label}...`

---

## 📝 Common LaTeX Examples

### Tables

```latex
\begin{table}[ht]
    \centering
    \caption{Your Table Caption}
    \label{tab:your_label}
    \begin{tabular}{@{}lcc@{}}
        \toprule
        \textbf{Column 1} & \textbf{Column 2} & \textbf{Column 3} \\
        \midrule
        Row 1 Data & 123 & Yes \\
        Row 2 Data & 456 & No \\
        \bottomrule
    \end{tabular}
\end{table}
```

### Equations

```latex
% Inline equation
The formula $E = mc^2$ is famous.

% Display equation
\begin{equation}
    f(x) = \int_{0}^{\infty} e^{-x^2} dx
    \label{eq:your_label}
\end{equation}
```

### Algorithms

```latex
\begin{algorithm}[H]
\caption{Your Algorithm Name}
\begin{algorithmic}[1]
\Require Input parameters
\Ensure Output result
\State Initialize variables
\For{each item in list}
    \If{condition}
        \State Do something
    \EndIf
\EndFor
\State \Return result
\end{algorithmic}
\end{algorithm}
```

### Code Listings

```latex
\begin{lstlisting}[language=Python, caption={Your Code}]
def hello_world():
    print("Hello, World!")
    return True
\end{lstlisting}
```

**Supported languages:** Python, Java, CSharp, JavaScript, SQL, HTML, PHP, C, C++

### Lists

```latex
% Numbered list
\begin{enumerate}
    \item First item
    \item Second item
\end{enumerate}

% Bullet list
\begin{itemize}
    \item First point
    \item Second point
\end{itemize}
```

---

## 📐 Formatting Specifications

| Element | Specification |
|---------|---------------|
| Paper Size | A4 |
| Font | Times New Roman, 12pt |
| Line Spacing | 1.5 (onehalfspacing) |
| Left Margin | 1.5 inches (for binding) |
| Other Margins | 1 inch |
| Front Matter Pages | Roman numerals (i, ii, iii...) |
| Main Content Pages | Arabic numerals (1, 2, 3...) |

---

## 🔧 Class Options

Change options in the `\documentclass` line:

```latex
% Line spacing options
\documentclass[singlespacing]{uokthesis}    % Single spacing
\documentclass[onehalfspacing]{uokthesis}   % 1.5 spacing (default)
\documentclass[doublespacing]{uokthesis}    % Double spacing

% Draft mode (shows overfull boxes)
\documentclass[draft]{uokthesis}
```

---

## ❓ Troubleshooting

### "Bibliography not appearing"
Run the full compilation sequence:
```bash
pdflatex thesis → bibtex thesis → pdflatex thesis → pdflatex thesis
```

### "Figure not found"
- Check the file path and name (case-sensitive!)
- Ensure the image is in the correct `figures/` folder
- Try removing the file extension from `\includegraphics`

### "Undefined citation"
- Make sure the citation key exists in `references.bib`
- Check for typos in the key name
- Run bibtex and pdflatex again

### "Missing package"
Install the package:
```bash
# TeX Live (Linux/Mac)
tlmgr install packagename

# MiKTeX (Windows)
mpm --install packagename
```

### "Overleaf compilation timeout"
- Split large chapters into smaller files
- Reduce image file sizes (compress PNGs)
- Use PDF format for diagrams

---

## 📚 Resources

- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX) - Comprehensive LaTeX guide
- [Overleaf Documentation](https://www.overleaf.com/learn) - Tutorials and examples
- [TeX Stack Exchange](https://tex.stackexchange.com) - Q&A community
- [Tables Generator](https://www.tablesgenerator.com) - Create LaTeX tables easily
- [BibTeX Entry Generator](https://truben.no/latex/bibtex/) - Generate citations

---

## 📞 Support

**For template issues:**
- Check this README first
- Search [TeX Stack Exchange](https://tex.stackexchange.com)

**For thesis content and guidelines:**
- Consult your supervisor
- Refer to Software Engineering Teaching Unit guidelines

---

## 📄 Version History

- **v1.0** (2024) - Initial release

---

**Good luck with your thesis!** 🎓🇱🇰
