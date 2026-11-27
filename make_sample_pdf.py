"""Creates samples/sample.pdf so you can try the app right away."""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

PAGES = [
    ("The Solar System",
     "The Solar System formed about 4.6 billion years ago from a giant cloud of gas and dust. "
     "The Sun contains more than 99 percent of the total mass of the Solar System. "
     "There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus and Neptune."),
    ("The Planets",
     "Jupiter is the largest planet in the Solar System. It is a gas giant with a famous storm called the Great Red Spot. "
     "Mars is called the Red Planet because iron oxide on its surface gives it a reddish color. "
     "Venus is the hottest planet, with surface temperatures around 465 degrees Celsius, because of its thick atmosphere."),
    ("Moons",
     "Earth has one moon. Jupiter has more than 90 known moons, including Ganymede, the largest moon in the Solar System. "
     "Saturn is famous for its rings, which are made mostly of ice particles."),
]

def main():
    out = Path(__file__).parent / "samples" / "sample.pdf"
    out.parent.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    story = []
    for i, (title, body) in enumerate(PAGES):
        if i:
            story.append(PageBreak())
        story += [Paragraph(title, styles["Title"]), Paragraph(body, styles["BodyText"])]
    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    print("Wrote", out)

if __name__ == "__main__":
    main()
