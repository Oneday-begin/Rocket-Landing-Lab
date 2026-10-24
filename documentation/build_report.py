from pathlib import Path
import json
import re
import textwrap
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                               PageBreak, Table, TableStyle, Image, Preformatted, KeepTogether)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from pypdf import PdfReader

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent.parent)
parser.add_argument('--source',type=Path)
parser.add_argument('--output',type=Path)
args=parser.parse_args()
root=args.project_root.resolve()
source=args.source or root/'documentation/placement-report.md'
pdf=args.output or root/'docs/Rocket-Landing-Placement-Report.pdf'
pdf.parent.mkdir(parents=True,exist_ok=True)
W,H=A4
width=W-100
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=10.5,leading=14,spaceAfter=7,textColor=colors.HexColor('#17212b')))
styles.add(ParagraphStyle(name='Chapter',fontName='Helvetica-Bold',fontSize=21,leading=26,spaceAfter=16,textColor=colors.black,keepWithNext=True))
styles.add(ParagraphStyle(name='Sub',fontName='Helvetica-Bold',fontSize=11.5,leading=14,spaceBefore=7,spaceAfter=4,textColor=colors.black,keepWithNext=True))
styles.add(ParagraphStyle(name='TableBody',fontName='Helvetica',fontSize=9,leading=11,spaceAfter=0))
styles.add(ParagraphStyle(name='TableHead',parent=styles['TableBody'],fontName='Helvetica-Bold',textColor=colors.black))
styles.add(ParagraphStyle(name='CodeReport',fontName='Courier-Bold',fontSize=8.5,leading=11,backColor=colors.HexColor('#f2f4f7'),borderPadding=4,spaceBefore=0,spaceAfter=0))
styles.add(ParagraphStyle(name='CaptionReport',fontName='Helvetica-Oblique',fontSize=9,leading=12,spaceAfter=10,textColor=colors.HexColor('#435466')))
styles.add(ParagraphStyle(name='CoverTitle',fontName='Helvetica-Bold',fontSize=32,leading=38,spaceAfter=18,textColor=colors.black))

def clean(text):
    return text.replace('→',' -> ').replace('‑','-').replace('—','-').replace('–','-').replace('’',"'").replace('“','"').replace('”','"').replace('×','x').replace('·','|')
def inline(text):
    text=escape(clean(text))
    text=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    return re.sub(r'(https?://[^\s]+)',lambda m:'<link href="'+m.group(1)+'" color="#075985">'+m.group(1)+'</link>',text)
def p(text,style='BodyReport'): return Paragraph(inline(text),styles[style])
def table(rows, widths=None):
    if widths is None:
        n=len(rows[0]); widths=([.30,.70] if n==2 else [.23,.23,.54] if n==3 else [1/n]*n)
    data=[[p(str(c),'TableHead' if i==0 else 'TableBody') for c in row] for i,row in enumerate(rows)]
    t=Table(data,colWidths=[width*x for x in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dbeaf3')),
                          ('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),
                          ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),
                          ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),
                          ('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f8fafc')])]))
    return [t,Spacer(1,12)]

class Report(BaseDocTemplate):
    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and f.style.name=='Chapter':
            text=f.getPlainText(); key='chapter-'+text.split(' ')[0]
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(text,key,0,False)
            self.notify('TOCEntry',(0,text,self.page,key))

def footer(c,doc):
    if doc.page>1:
        c.setFont('Helvetica',8)
        c.setFillColor(colors.HexColor('#435466'))
        c.drawString(50,30,'Rocket Landing Lab | Placement preparation')
        c.drawRightString(W-50,30,str(doc.page))

doc=Report(str(pdf),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=46,bottomMargin=48,
           title='Rocket Landing Lab Technical Report and Placement Preparation',author='Divyansh')
doc.addPageTemplates(PageTemplate(id='main',frames=[Frame(50,48,width,H-94,id='normal',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)],onPage=footer))

def architecture():
    d=Drawing(width,210)
    rows=[('TRAIN','Environment and wrappers','PPO actor and critic','Checkpoints and validation'),
          ('EVALUATE','Saved policy and weather','Original reward rollouts','CSV JSON and reports'),
          ('CONTROL','Tkinter settings','Owned worker process','Queue and progress display')]
    for i,(label,*boxes) in enumerate(rows):
        y=160-i*65
        d.add(String(0,y+30,label,fontName='Helvetica-Bold',fontSize=9,fillColor=colors.black))
        for j,text in enumerate(boxes):
            x=j*167
            d.add(Rect(x,y-8,155,28,fillColor=colors.HexColor('#e9f2f8'),strokeColor=colors.HexColor('#b2c7d6'),rx=4))
            d.add(String(x+77.5,y+3,text,fontName='Helvetica',fontSize=8.1,textAnchor='middle'))
            if j<2:
                d.add(Line(x+155,y+6,x+164,y+6,strokeColor=colors.HexColor('#435466')))
                d.add(Polygon([x+164,y+6,x+160,y+9,x+160,y+3],fillColor=colors.HexColor('#435466')))
    return [d,p('Figure 1  The three execution paths share the same model and evaluation modules.','CaptionReport')]

batch=next((root/'artifacts/quality').glob('*/quality-gate.json')).parent
def special(line):
    if line=='@@TOC':
        toc=TableOfContents(); toc.levelStyles=[ParagraphStyle(name='TOC',fontName='Helvetica',fontSize=8.7,leading=10,spaceBefore=0)]
        return [PageBreak(),p('Contents','Sub'),toc]
    if line=='@@ARCH': return architecture()
    if line=='@@WEB_PREVIEW':
        return [Image(str(root/'docs/web-lab-preview.png'),width=width,height=width*712/1265),p('Figure 4  Guided introduction before the analytics and experiment workspace.','CaptionReport')]
    if line=='@@ORIGINAL_TABLE':
        rows=[['Policy','Mean score','Std dev','Landed','Timeout']]
        for preset in ('baseline','time','hover','fixed'):
            v=json.loads((root/f'runs/{preset}/seed_0/evaluation.json').read_text())
            rows.append([preset,f"{v['mean_base_return']:.2f}",f"{v['std_base_return']:.2f}",f"{v['landed_rate']:.0%}",f"{v['timeout_rate']:.0%}"])
        return table(rows,[.23,.22,.19,.18,.18])
    if line=='@@WEATHER_TABLE':
        cases=json.loads((batch/'results.json').read_text())
        gates=json.loads((batch/'quality-gate.json').read_text())['models']
        rows=[['Policy','Calm landed','Breezy landed','Gusty landed','Gate']]
        for preset in ('baseline','fixed','hover','time'):
            selected=[r for r in cases if '/'+preset+'/' in r['model_path'].replace('\\','/')]
            gate=next(m for m in gates if f'. {preset} /' in m['model'])
            rows.append([preset]+[f"{next(r for r in selected if r['scenario']==s)['landed_rate']:.0%}" for s in ('calm','breezy','gusty')]+['PASS' if gate['passed'] else 'FAIL'])
        return table(rows,[.2,.2,.2,.2,.2])
    if line in ('@@LEARNING_CHART','@@WEATHER_CHART'):
        path=root/'artifacts/learning-curves.png' if line=='@@LEARNING_CHART' else batch/'weather-comparison.png'
        from PIL import Image as PILImage
        with PILImage.open(path) as im: h=width*im.height/im.width
        title=('Figure 2  Original validation scores and shaped training curves from the saved experiment.' if line=='@@LEARNING_CHART' else
               'Figure 3  The 600-flight weather comparison with landing uncertainty intervals and original scores.')
        return [KeepTogether([Image(str(path),width=width,height=h),p(title,'CaptionReport')])]
    raise ValueError(line)

lines=source.read_text(encoding='utf-8').splitlines()
story=[]; i=0
while i<len(lines):
    line=lines[i]
    if not line.strip(): i+=1; continue
    if line.startswith('@@'):
        story.extend(special(line)); i+=1;continue
    if line.startswith('```'):
        lang=line[3:]; code=[]; i+=1
        while i<len(lines) and not lines[i].startswith('```'):
            raw=clean(lines[i]);
            if len(raw)>96 and lang=='powershell' and not raw.startswith('#'):
                pieces=textwrap.wrap(raw,width=92,break_long_words=False,break_on_hyphens=False)
                code.extend([x+' `' for x in pieces[:-1]]+[pieces[-1]])
            else: code.append(raw)
            i+=1
        # Paragraph-level preformatted blocks permit splitting over page boundaries.
        for raw in code:
            story.append(Preformatted(raw or ' ',styles['CodeReport'],maxLineLength=100))
        i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[x.strip() for x in lines[i].strip('|').split('|')]
            if not all(re.match(r'^[-: ]+$',x) for x in cells):rows.append(cells)
            i+=1
        story.extend(table(rows));continue
    if line.startswith('# '):
        text=line[2:]
        if re.match(r'^\d+ ',text):
            story.append(PageBreak());story.append(p(text,'Chapter'))
        else:
            story.extend([Spacer(1,30),p(text,'CoverTitle')])
        i+=1;continue
    if line.startswith('## '):story.append(p(line[3:],'Sub'));i+=1;continue
    if line.startswith('- '):story.append(Paragraph(inline(line[2:]),styles['BodyReport'],bulletText='\u2022'));i+=1;continue
    paragraph=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','@@','```','- ')):
        # Keep numbered procedure steps separate.
        if re.match(r'^\d+\. ',lines[i]):break
        paragraph.append(lines[i]);i+=1
    story.append(p(' '.join(paragraph)))

doc.multiBuild(story)
reader=PdfReader(pdf)
text='\n'.join(page.extract_text() or '' for page in reader.pages)
assert len(reader.pages)>=27
assert '@@' not in text
assert '9,797' in text and '238.80' in text
print(json.dumps({'pdf':str(pdf),'pages':len(reader.pages),'words':len(text.split()),'bytes':pdf.stat().st_size}))
