%% uokthesis.cls
%% University of Kelaniya BSc Honours Thesis Template
%% Faculty of Science - Software Engineering
%% 
%% This template follows the official thesis formatting guidelines
%% of the University of Kelaniya, Sri Lanka.
%%
%% Author: University of Kelaniya - Software Engineering Teaching Unit
%% Version: 1.0
%% Last Updated: 2024
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uokthesis}[2024/01/01 University of Kelaniya BSc Honours Thesis Class]

%% ============================================================================
%% CLASS OPTIONS
%% ============================================================================

% Draft/Final mode
\newif\if@draftmode
\@draftmodefalse
\DeclareOption{draft}{\@draftmodetrue\PassOptionsToClass{draft}{report}}
\DeclareOption{final}{\@draftmodefalse\PassOptionsToClass{final}{report}}

% Spacing options
\newif\if@singlespacing
\newif\if@onehalfspacing
\newif\if@doublespacing
\@singlespacingfalse
\@onehalfspacingtrue
\@doublespacingfalse
\DeclareOption{singlespacing}{\@singlespacingtrue\@onehalfspacingfalse\@doublespacingfalse}
\DeclareOption{onehalfspacing}{\@singlespacingfalse\@onehalfspacingtrue\@doublespacingfalse}
\DeclareOption{doublespacing}{\@singlespacingfalse\@onehalfspacingfalse\@doublespacingtrue}

% Pass unknown options to report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax

%% ============================================================================
%% BASE CLASS
%% ============================================================================

\LoadClass[12pt,a4paper,oneside]{report}

%% ============================================================================
%% REQUIRED PACKAGES
%% ============================================================================

% Page geometry
\RequirePackage[
    a4paper,
    left=1.5in,     % Binding margin
    right=1in,
    top=1in,
    bottom=1in,
    headheight=14pt
]{geometry}

% Font packages
\RequirePackage{times}          % Times New Roman font
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}

% Spacing
\RequirePackage{setspace}

% Graphics
\RequirePackage{graphicx}
\RequirePackage{subcaption}
\RequirePackage{float}

% Tables
\RequirePackage{booktabs}
\RequirePackage{longtable}
\RequirePackage{multirow}
\RequirePackage{array}
\RequirePackage{tabularx}

% Math
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}

% Algorithms
\RequirePackage{algorithm}
\RequirePackage{algpseudocode}

% Code listings
\RequirePackage{listings}

% Colors
\RequirePackage{xcolor}

% References and citations
\RequirePackage[numbers,sort&compress]{natbib}
\RequirePackage[hidelinks]{hyperref}

% Headers and footers
\RequirePackage{fancyhdr}

% For title formatting
\RequirePackage{titlesec}

% For table of contents formatting
\RequirePackage{tocloft}

% Sample text (for template demonstration)
\RequirePackage{lipsum}

% For appendices
\RequirePackage[toc,page,title]{appendix}

% For URLs
\RequirePackage{url}

% For better list formatting
\RequirePackage{enumitem}

%% ============================================================================
%% SPACING CONFIGURATION
%% ============================================================================

\if@singlespacing
    \singlespacing
\fi
\if@onehalfspacing
    \onehalfspacing
\fi
\if@doublespacing
    \doublespacing
\fi

%% ============================================================================
%% LIST CONFIGURATION
%% ============================================================================

% Configure itemize and enumerate for proper indentation
% Ensure wrapped text doesn't overflow into left margin
\setlist[itemize]{
    leftmargin=2.5em,
    labelsep=0.5em,
    itemsep=0.5ex,
    parsep=0.5ex
}
\setlist[enumerate]{
    leftmargin=2.5em,
    labelsep=0.5em,
    itemsep=0.5ex,
    parsep=0.5ex
}

%% ============================================================================
%% CUSTOM COMMANDS - THESIS INFORMATION
%% ============================================================================

\newcommand{\thesistitle}[1]{\gdef\@thesistitle{#1}}
\newcommand{\groupnumber}[1]{\gdef\@groupnumber{#1}}
\newcommand{\studentone}[3]{\gdef\@studentonename{#1}\gdef\@studentoneno{#2}\gdef\@studentonerole{#3}}
\newcommand{\studenttwo}[3]{\gdef\@studenttwoname{#1}\gdef\@studenttwono{#2}\gdef\@studenttworole{#3}}
\newcommand{\studentthree}[3]{\gdef\@studentthreename{#1}\gdef\@studentthreeno{#2}\gdef\@studentthreerole{#3}}
\newcommand{\studentfour}[3]{\gdef\@studentfourname{#1}\gdef\@studentfourno{#2}\gdef\@studentfourrole{#3}}

\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\faculty}[1]{\gdef\@faculty{#1}}
\newcommand{\department}[1]{\gdef\@department{#1}}
\newcommand{\academicyear}[1]{\gdef\@academicyear{#1}}
\newcommand{\supervisorname}[1]{\gdef\@supervisorname{#1}}
\newcommand{\supervisortitle}[1]{\gdef\@supervisortitle{#1}}
\newcommand{\supervisordetails}[1]{\gdef\@supervisordetails{#1}}
\newcommand{\submissiondate}[1]{\gdef\@submissiondate{#1}}

% Default values
\thesistitle{Research Title}
\groupnumber{Group 07}
\studentone{Student One}{SE/20XX/001}{Research Lead}
\studenttwo{Student Two}{SE/20XX/002}{Lead Developer}
\studentthree{Student Three}{SE/20XX/003}{Project Manager}
\studentfour{Student Four}{SE/20XX/004}{Documentations Lead}
\degree{Bachelor of Science Honours in Software Engineering}
\faculty{Faculty of Science}
\department{Software Engineering Teaching Unit}
\academicyear{20XX/20XX}
\supervisorname{Supervisor Name}
\supervisortitle{Lecturer}
\supervisordetails{Software Engineering Teaching Unit,\\Faculty of Science,\\University of Kelaniya,\\Sri Lanka}
\submissiondate{\today}

%% ============================================================================
%% TITLE PAGE
%% ============================================================================

\newcommand{\maketitlepage}{%
    \begin{titlepage}
        \centering
        \vspace*{0.5cm}
        
        % University Logo
        \includegraphics[width=3.5cm]{images/uok_logo.png}\\[1.0cm]
        
        % Thesis Title
        {\Large\bfseries\@thesistitle}\\[1.0cm]
        
        % Degree Statement
        {\large IN PARTIAL FULFILLMENT FOR THE DEGREE OF}\\[0.2cm]
        {\large\MakeUppercase{\@degree}}\\[0.2cm]
        {\large\MakeUppercase{\@faculty}}\\[0.2cm]
        {\large UNIVERSITY OF KELANIYA}\\[0.8cm]
        
        % Group Information
        {\large\bfseries Group Number: \@groupnumber}\\[0.5cm]
        
        % Student Information List (Centered formatted block)
        {\large
        \begin{tabular}{c@{\hspace{0.3cm}-\hspace{0.3cm}}l@{\hspace{0.3cm}-\hspace{0.3cm}}l}
            \@studentoneno & \@studentonename & \@studentonerole \\
            \@studenttwono & \@studenttwoname & \@studenttworole \\
            \@studentthreeno & \@studentthreename & \@studentthreerole \\
            \@studentfourno & \@studentfourname & \@studentfourrole \\
        \end{tabular}}\\[0.8cm]
        
        % Supervisor Information
        {\large\bfseries Academic Supervisor: \@supervisorname}\\[0.3cm]
        {\large \@academicyear}\\[0.8cm]
        
        \vfill
    \end{titlepage}
    \clearpage
}

%% ============================================================================
%% DECLARATION PAGE
%% ============================================================================

\newcommand{\makedeclaration}{%
    \chapter*{Declaration}
    \addcontentsline{toc}{chapter}{Declaration}
    
    We hereby declare that this thesis does not incorporate, without acknowledgement, any material previously submitted for a degree or diploma at any university. To the best of my knowledge and belief, it contains no material previously published or written by another person, except where due reference is made in the text.
   
    We further affirm that artificial intelligence-based tools were used only in a limited, responsible, and transparent manner during the preparation of this thesis. These tools were employed solely for language refinement, structural suggestions, and preliminary idea exploration. All substantive content, including the conceptualization, analysis, interpretation of results, and conclusions, is entirely our own. We take full responsibility for the accuracy, originality, and integrity of the work presented herein.
    
    \vspace{0.5cm}

    \noindent
    \textbf{Research Name:} \@thesistitle \\
    
    \vspace{0.4cm}
    \noindent
    \begin{tabular}{@{}p{4.5cm}l@{\hspace{0.5cm}}p{3cm}l}
        \textbf{Student Name} & \textbf{Student No} & \textbf{Signature} & \textbf{Date} \\ \midrule
        \@studentonename & \@studentoneno & \dotfill & \dotfill \\ [0.4cm]
        \@studenttwoname & \@studenttwono & \dotfill & \dotfill \\ [0.4cm]
        \@studentthreename & \@studentthreeno & \dotfill & \dotfill \\ [0.4cm]
        \@studentfourname & \@studentfourno & \dotfill & \dotfill \\
    \end{tabular}
    
    \vspace{1.0cm}
    
    I hereby certify that this report document is work done by these students under my supervision 
    and guidance.

    \vspace{0.4cm}

    \noindent
    \begin{tabular}{@{}l@{\hspace{0.5cm}}c@{\hspace{0.3cm}}p{8cm}}
        Supervisor Name    & : & \@supervisorname \\[0.3cm]
        Supervisor Details & : & \@supervisortitle, \\
                           \@supervisordetails \\[0.5cm]
        Signature of the Supervisor & : & \dotfill \\[0.5cm]
        Date               & : & \dotfill \\
    \end{tabular}
    
    \clearpage
}

%% ============================================================================
%% FRONT MATTER / MAIN MATTER / BACK MATTER
%% ============================================================================

\newcommand{\frontmatter}{%
    \cleardoublepage
    \pagenumbering{roman}
    \pagestyle{plain}
}

\newcommand{\mainmatter}{%
    \cleardoublepage
    \pagenumbering{arabic}
    \pagestyle{fancy}
}

\newcommand{\backmatter}{%
    \cleardoublepage
}

%% ============================================================================
%% CHAPTER AND SECTION FORMATTING
%% ============================================================================

% Chapter title formatting - Left aligned style
% Uses \chaptertitlename to automatically show "Chapter" or "Appendix"
\titleformat{\chapter}[display]
    {\normalfont\bfseries}
    {\large \chaptertitlename\ \thechapter}
    {0pt}
    {\LARGE}

% Unnumbered chapter formatting (for frontmatter, bibliography, etc.)
\titleformat{name=\chapter,numberless}[display]
    {\normalfont\bfseries}
    {}
    {0pt}
    {\LARGE}

\titlespacing*{\chapter}{0pt}{20pt}{30pt}

% Section formatting - left aligned block style
\titleformat{\section}[block]
    {\normalfont\Large\bfseries}
    {\thesection}
    {1em}
    {}
\titlespacing*{\section}{0pt}{3.5ex plus 1ex minus .2ex}{2.3ex plus .2ex}

% Subsection formatting - left aligned block style
\titleformat{\subsection}[block]
    {\normalfont\large\bfseries}
    {\thesubsection}
    {1em}
    {}
\titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}

% Subsubsection formatting - left aligned block style
\titleformat{\subsubsection}[block]
    {\normalfont\normalsize\bfseries}
    {\thesubsubsection}
    {1em}
    {}
\titlespacing*{\subsubsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}

%% ============================================================================
%% TABLE OF CONTENTS FORMATTING
%% ============================================================================

\setcounter{tocdepth}{3}
\setcounter{secnumdepth}{3}

% Chapter entries in TOC
\renewcommand{\cftchapfont}{\bfseries}
\renewcommand{\cftchappagefont}{\bfseries}
\renewcommand{\cftchapleader}{\cftdotfill{\cftdotsep}}

%% ============================================================================
%% HEADER AND FOOTER
%% ============================================================================

\fancyhf{}
\fancyhead[L]{\leftmark}
\fancyhead[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0pt}

% Redefine plain page style (used by chapter pages)
\fancypagestyle{plain}{%
    \fancyhf{}
    \fancyfoot[C]{\thepage}
    \renewcommand{\headrulewidth}{0pt}
    \renewcommand{\footrulewidth}{0pt}
}

%% ============================================================================
%% ENVIRONMENTS
%% ============================================================================

% Acknowledgement
\newenvironment{acknowledgement}{%
    \chapter*{Acknowledgement}
    \addcontentsline{toc}{chapter}{Acknowledgement}
}{\clearpage}

% Abstract
\renewenvironment{abstract}{%
    \chapter*{Abstract}
    \addcontentsline{toc}{chapter}{Abstract}
}{\clearpage}

% List of Abbreviations
\newenvironment{abbreviations}{%
    \chapter*{List of Abbreviations}
    \addcontentsline{toc}{chapter}{List of Abbreviations}
}{\clearpage}

%% ============================================================================
%% THEOREM ENVIRONMENTS
%% ============================================================================

\theoremstyle{definition}
\newtheorem{definition}{Definition}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{example}{Example}[chapter]
\newtheorem{remark}{Remark}[chapter]

%% ============================================================================
%% CODE LISTING STYLE
%% ============================================================================

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{codeblue}{rgb}{0,0,0.8}
\definecolor{backcolour}{rgb}{0.97,0.97,0.97}

\lstdefinestyle{codestyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{codeblue}\bfseries,
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    frame=single,
    framerule=0.5pt,
    rulecolor=\color{codegray}
}

\lstset{style=codestyle}

% Define C# language for listings
\lstdefinelanguage{CSharp}{
    keywords={abstract, as, base, bool, break, byte, case, catch, char, checked, class, const, continue, decimal, default, delegate, do, double, else, enum, event, explicit, extern, false, finally, fixed, float, for, foreach, goto, if, implicit, in, int, interface, internal, is, lock, long, namespace, new, null, object, operator, out, override, params, private, protected, public, readonly, ref, return, sbyte, sealed, short, sizeof, stackalloc, static, string, struct, switch, this, throw, true, try, typeof, uint, ulong, unchecked, unsafe, ushort, using, virtual, void, volatile, while},
    sensitive=true,
    morecomment=[l]{//},
    morecomment=[s]{/*}{*/},
    morestring=[b]",
    morestring=[b]',
}

%% ============================================================================
%% HYPERREF CONFIGURATION
%% ============================================================================

\hypersetup{
    colorlinks=false,
    linkcolor=black,
    filecolor=black,
    urlcolor=blue,
    citecolor=black,
    pdftitle={\@thesistitle},
    pdfauthor={\@studentonename, \@studenttwoname, \@studentthreename, \@studentfourname},
    pdfsubject={BSc Honours Thesis},
    pdfkeywords={University of Kelaniya, Thesis}
}

%% ============================================================================
%% HELPER COMMANDS
%% ============================================================================

% Set graphics path for chapters
\newcommand{\chapterfigpath}[1]{%
    \graphicspath{{chapters/#1/figures/}{images/}}
}

% Command for including figures
\newcommand{\insertfigure}[4][ht]{%
    % #1 = placement (optional, default ht)
    % #2 = filename
    % #3 = caption
    % #4 = label
    \begin{figure}[#1]
        \centering
        \includegraphics[width=0.8\textwidth]{#2}
        \caption{#3}
        \label{fig:#4}
    \end{figure}
}

% Command for including tables
\newcommand{\inserttable}[3]{%
    % #1 = table content file
    % #2 = caption
    % #3 = label
    \input{#1}
}

%% ============================================================================
%% RENAME DEFAULT HEADINGS
%% ============================================================================

% Rename default headings to title case
\renewcommand{\contentsname}{Table of Contents}
\renewcommand{\listfigurename}{List of Figures}
\renewcommand{\listtablename}{List of Tables}
\renewcommand{\bibname}{Bibliography}

% Rename algorithm list heading
\renewcommand{\listalgorithmname}{List of Algorithms}

% Appendix formatting - show "Appendix A" above title
\renewcommand{\appendixname}{Appendix}

%% ============================================================================
%% END OF CLASS FILE
%% ============================================================================

\endinput
