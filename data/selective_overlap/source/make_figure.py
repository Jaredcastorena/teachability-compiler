from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, white

OUT = Path(__file__).parent / 'figures' / 'architecture.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
c = canvas.Canvas(str(OUT), pagesize=(500, 245))
c.setTitle('Selective overlap records before a destructive transition')
navy = HexColor('#19374d')
teal = HexColor('#116b70')
muted = HexColor('#586975')

def box(x,y,w,h,lines,color=navy,fill='#f0f4f6'):
    c.setFillColor(HexColor(fill)); c.setStrokeColor(color); c.setLineWidth(0.9)
    c.roundRect(x,y,w,h,6,stroke=1,fill=1)
    c.setFillColor(color)
    total=len(lines)*12
    for i,line in enumerate(lines):
        c.setFont('Helvetica-Bold' if i==0 else 'Helvetica',9 if i==0 else 8.5)
        c.drawCentredString(x+w/2,y+h/2+total/2-10-i*12,line)

def arrow(x1,y1,x2,y2,color=navy):
    import math
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(1.1)
    c.line(x1,y1,x2,y2)
    a=math.atan2(y2-y1,x2-x1); s=5
    p=c.beginPath(); p.moveTo(x2,y2)
    p.lineTo(x2-s*math.cos(a-.45),y2-s*math.sin(a-.45))
    p.lineTo(x2-s*math.cos(a+.45),y2-s*math.sin(a+.45));p.close()
    c.drawPath(p,stroke=0,fill=1)

box(12,159,124,49,['Layer input','h_l'])
box(188,159,132,49,['Ordinary transition','projection / activation'])
box(371,159,117,49,['Next local state','h_(l+1)'])
box(183,64,142,65,['Selected overlap record','CReLU(U_l h_l)','captured before collapse'],teal,'#edf7f6')
box(371,64,117,65,['Atlas reconstruction','maps + layer IDs','retained constraints'],teal,'#edf7f6')
arrow(136,184,188,184);arrow(320,184,371,184)
arrow(74,159,74,97,teal);arrow(74,97,183,97,teal)
arrow(325,97,371,97,teal);arrow(429,159,429,129,teal)
arrow(429,64,429,38,teal)
c.setFillColor(navy);c.setFont('Helvetica-Bold',10);c.drawCentredString(429,22,'Task readout')
c.setFillColor(muted);c.setFont('Helvetica',8)
c.drawString(12,226,'The protected record bypasses the operation that would erase its information.')
c.drawString(12,16,'Selection and gluing require a target-sufficiency argument; agreement alone is insufficient.')
c.showPage();c.save()
print(OUT)
