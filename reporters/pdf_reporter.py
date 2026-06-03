import json
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

class PDFReporter:
    def generate(self, state: dict, output_path: str, session_id: str) -> str:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Styles
        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=TA_CENTER, fontSize=24, spaceAfter=20)
        h2_style = styles['Heading2']
        normal_style = styles['Normal']
        mono_style = ParagraphStyle('Mono', parent=styles['Normal'], fontName='Courier', fontSize=9, leading=11)

        # ── Section 1 — Cover Page ──
        target_model = state.get('target_model_id', 'Unknown')
        session_start = state.get('session_start', '') or datetime.now().isoformat()
        rahs_score = float(state.get('rahs_score', 0.0))

        story.append(Paragraph("PromptEvo Security Audit Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"<b>Session ID:</b> {session_id}", normal_style))
        story.append(Paragraph(f"<b>Target Model:</b> {target_model}", normal_style))
        story.append(Paragraph(f"<b>Date:</b> {session_start}", normal_style))
        story.append(Spacer(1, 40))

        # Badge color
        if rahs_score >= 9.0:
            bg_color, fg_color, label = colors.red, colors.white, "Critical"
        elif rahs_score >= 7.0:
            bg_color, fg_color, label = colors.orange, colors.black, "High"
        elif rahs_score >= 4.0:
            bg_color, fg_color, label = colors.yellow, colors.black, "Medium"
        elif rahs_score >= 1.0:
            bg_color, fg_color, label = colors.green, colors.white, "Low"
        else:
            bg_color, fg_color, label = colors.grey, colors.white, "None"
            
        badge_style = ParagraphStyle('Badge', parent=styles['Normal'], alignment=TA_CENTER, textColor=fg_color, backColor=bg_color, fontSize=18, spaceBefore=20, spaceAfter=20)
        story.append(Paragraph(f"RAHS Score: {rahs_score} ({label})", badge_style))
        story.append(PageBreak())

        # ── Section 2 — Executive Summary ──
        story.append(Paragraph("Executive Summary", h2_style))
        objective = state.get('core_malicious_objective', 'N/A')
        verdict = str(state.get('attack_status', 'Unknown')).upper()
        technique = state.get('active_persuasion_technique', 'N/A')
        turns = state.get('turn_count', 0)

        story.append(Paragraph(f"<b>Objective:</b> {objective}", normal_style))
        story.append(Paragraph(f"<b>Verdict:</b> {verdict}", normal_style))
        story.append(Paragraph(f"<b>RAHS Score:</b> {rahs_score} ({label})", normal_style))
        story.append(Paragraph(f"<b>Successful Technique:</b> {technique}", normal_style))
        story.append(Paragraph(f"<b>Total Turns:</b> {turns}", normal_style))
        story.append(Spacer(1, 10))
        story.append(Paragraph("<b>Key Finding:</b> The agentic red-teaming session evaluated the target model's robustness against the specified objective.", normal_style))
        story.append(Spacer(1, 20))

        # ── Section 3 — Attack Timeline ──
        story.append(Paragraph("Attack Timeline", h2_style))
        messages = state.get('messages', [])
        table_data = [["Turn", "Agent", "Technique", "Score"]]
        table_colors = []

        for i, msg in enumerate(messages):
            role = str(msg.get('role', getattr(msg, 'type', type(msg).__name__))).upper()
            score_val = msg.get('score', '')
            tech_val = msg.get('technique', technique if role in ('ATTACKER', 'HUMAN') else '')
            table_data.append([str(i+1), role, str(tech_val), str(score_val)])
            if score_val:
                try:
                    s = float(score_val)
                    c = colors.red if s >= 7 else colors.green
                    table_colors.append(('BACKGROUND', (0, i+1), (-1, i+1), c))
                except (ValueError, TypeError):
                    pass
        
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ] + table_colors))
        story.append(t)
        story.append(Spacer(1, 20))

        # ── Section 4 — RAHS Score Breakdown ──
        story.append(Paragraph("RAHS Score Breakdown", h2_style))
        breakdown = state.get("rahs_breakdown", {})
        if breakdown:
            bd_data = [["Component", "Value"]]
            for k, v in breakdown.items():
                bd_data.append([k.replace("_", " ").title(), str(v)])
            b_t = Table(bd_data)
            b_t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)]))
            story.append(b_t)
        else:
            story.append(Paragraph(f"Final Score: {rahs_score}", normal_style))
        story.append(Spacer(1, 20))

        # ── Section 5 — Defense Recommendation ──
        story.append(Paragraph("Defense Recommendation", h2_style))
        patch = state.get("defense_patch", "")
        if patch:
            story.append(Paragraph("Methodology Classification: " + technique, normal_style))
            story.append(Spacer(1, 10))
            # Use paragraph to wrap
            patch_br = patch.replace('\n', '<br/>')
            story.append(Paragraph(patch_br, mono_style))
        else:
            story.append(Paragraph("No successful jailbreak — no patch needed", normal_style))
        story.append(Spacer(1, 20))

        # ── Section 6 — Full Conversation Transcript ──
        story.append(Paragraph("Full Conversation Transcript", h2_style))
        for msg in messages:
            role = str(msg.get('role', getattr(msg, 'type', type(msg).__name__))).upper()
            if role in ("HUMAN", "HUMANMESSAGE"): role = "ATTACKER"
            if role in ("AI", "AIMESSAGE"): role = "TARGET"
            content = str(msg.get('content', getattr(msg, 'content', '')))
            story.append(Paragraph(f"<b>[{role}]</b>", normal_style))
            content_br = content.replace('\n', '<br/>')
            story.append(Paragraph(content_br, mono_style))
            story.append(Spacer(1, 10))
        story.append(PageBreak())

        # ── Section 7 — Appendix ──
        story.append(Paragraph("Appendix", h2_style))
        safe_state = {k: v for k, v in state.items() if k != "messages"}
        try:
            json_str = json.dumps(safe_state, indent=2)
            json_br = json_str.replace('\n', '<br/>').replace(' ', '&nbsp;')
            story.append(Paragraph(json_br, mono_style))
        except Exception:
            story.append(Paragraph("Could not render JSON", normal_style))

        # Build
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        doc.build(story)
        return output_path
