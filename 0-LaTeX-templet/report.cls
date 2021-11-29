\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{report}[2021/09/23 v1.0 UCAS Multi-modal-analysis-and-understand Thesis By Ruyun]

\LoadClassWithOptions{ctexart}
\RequirePackage[a4paper, top=2cm, bottom=2.5cm, left=2cm, right=2cm]{geometry}


\pagestyle{plain}


\RequirePackage{caption}
\RequirePackage{setspace}
\RequirePackage{titlesec}
\RequirePackage{graphicx}
\RequirePackage{CJKfntef}
\RequirePackage{newclude}
\RequirePackage{tikz}
%\RequirePackage{enumerate}
\RequirePackage{enumitem}
\RequirePackage{amsmath}
\RequirePackage[title]{appendix}
\RequirePackage{pythonhighlight}
\RequirePackage{xcolor}
\RequirePackage{tcolorbox}
\RequirePackage{lipsum}
\RequirePackage{subfigure}
\RequirePackage{float}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{booktabs}


\graphicspath{{figures/}}

\DeclareRobustCommand{\stuname}{}
\DeclareRobustCommand{\stuno}{}
\DeclareRobustCommand{\stuinfo}[2]{
	\renewcommand{\stuname}{#1}
	\renewcommand{\stuno}{#2}
}

\DeclareRobustCommand{\coursename}{}
\DeclareRobustCommand{\courseno}{}
\DeclareRobustCommand{\courseinfo}[2]{
	\renewcommand{\coursename}{#1}
	\renewcommand{\courseno}{#2}
}

\DeclareRobustCommand{\content}{}
\DeclareRobustCommand{\workcontent}[1]{
	\renewcommand{\content}{#1}
}

\DeclareRobustCommand{\year}{}
\DeclareRobustCommand{\mon}{}
\DeclareRobustCommand{\day}{}
\DeclareRobustCommand{\date}[3]{
	\renewcommand{\year}{#1}
	\renewcommand{\mon}{#2}
	\renewcommand{\day}{#3}
}





