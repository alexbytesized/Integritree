import { useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { X } from "lucide-react"
import "./InfoModal.css"

const fraudTerms = {
  TP: ["TP (True Positive)", "A fraudulent transaction correctly classified as fraudulent."],
  TN: ["TN (True Negative)", "A legitimate transaction correctly classified as legitimate."],
  FP: ["FP (False Positive)", "A legitimate transaction incorrectly classified as fraudulent."],
  FN: ["FN (False Negative)", "A fraudulent transaction incorrectly classified as legitimate."],
}

const topics = {
  Precision: {
    title: "Precision",
    definition: "Precision answers “of the transactions this model predicted as fraud, how many were actually labeled fraud?”",
    formula: "Precision = TP / (TP + FP)",
    where: [fraudTerms.TP, fraudTerms.FP],
    interpretation: "A higher value means a greater proportion of fraud predictions were correct. The value ranges from 0 to 1, or 0% to 100%.",
  },
  Recall: {
    title: "Recall",
    definition: "Recall answers “of the transactions actually labeled fraud, how many did this model identify as fraud?”",
    formula: "Recall = TP / (TP + FN)",
    where: [fraudTerms.TP, fraudTerms.FN],
    interpretation: "A higher value means a greater share of actual fraud cases was detected. The value ranges from 0 to 1, or 0% to 100%.",
  },
  "F1-Score": {
    title: "F1-Score",
    definition: "F1 combines precision and recall into their harmonic mean, giving a single measure of the balance between reliable fraud predictions and fraud detection coverage.",
    formula: "F1 = 2 * TP / (2 * TP + FP + FN)",
    where: [fraudTerms.TP, fraudTerms.FP, fraudTerms.FN],
    interpretation: "A higher value indicates a stronger balance between precision and recall. The value ranges from 0 to 1, or 0% to 100%.",
  },
  MCC: {
    title: "Matthews Correlation Coefficient",
    definition: "MCC measures association between predicted and actual labels using all four confusion-matrix outcomes. It is useful when fraud and legitimate counts differ greatly.",
    formula: "MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))",
    where: [fraudTerms.TP, fraudTerms.TN, fraudTerms.FP, fraudTerms.FN],
    interpretation: [
      "+1: perfect agreement with the actual labels.",
      "0: no correlation between predicted and actual labels, where the coefficient is defined.",
      "-1: complete inverse agreement in a nondegenerate binary evaluation.",
    ],
  },
  "PR-AUC": {
    title: "Area Under the Precision-Recall Curve",
    definition: "PR-AUC summarizes the trade-off between precision and recall as the fraud-score threshold changes, instead of measuring only one classification threshold.",
    formula: "PR-AUC = sumᵢ (Rᵢ - Rᵢ₋₁) * Pᵢ",
    where: [["Rᵢ", "Recall at successive thresholds."], ["Pᵢ", "Precision at successive thresholds."]],
    interpretation: "Higher values indicate stronger precision-recall performance on the same evaluation population. The value ranges from 0 to 1, or 0% to 100%.",
  },
  "Percentage Difference": {
    title: "Percentage Difference",
    definition: "Percentage Difference compares RF-SMOTE with benchmark RF using the average of their metric values as the reference, while keeping the direction of the difference.",
    formula: "Percentage Difference = 100 * (A - B) / ((A + B) / 2)",
    where: [["A", "RF-SMOTE metric score."], ["B", "Benchmark RF metric score."]],
    interpretation: [
      "Positive: RF-SMOTE has the higher metric value.",
      "Negative: benchmark RF has the higher metric value.",
      "Zero: equal metric values.",
    ],
  },
  "McNemar's Test": {
    title: "McNemar's Test",
    definition: "McNemar's Test evaluates whether the two models have different classification error rates when they are evaluated on the same labeled transactions.",
    formula: "χ² = (abs(B - C) - 1)² / (B + C)",
    where: [
      ["B", "Count where benchmark RF is correct and RF-SMOTE is incorrect."],
      ["C", "Count where benchmark RF is incorrect and RF-SMOTE is correct."],
    ],
    interpretationIntro: "Compare the p-value with the declared significance level, alpha = 0.05:",
    interpretation: [
      "p < alpha: reject the null hypothesis of equal error rates.",
      "p >= alpha: fail to reject the null hypothesis.",
    ],
  },
}

const InfoModal = ({ topic, onClose }) => {
  const closeButtonRef = useRef(null)
  const content = topics[topic]

  useEffect(() => {
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = "hidden"
    closeButtonRef.current?.focus()

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault()
        onClose()
      } else if (event.key === "Tab") {
        // The close button is the dialog's only interactive element.
        event.preventDefault()
        closeButtonRef.current?.focus()
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => {
      document.removeEventListener("keydown", handleKeyDown)
      document.body.style.overflow = previousOverflow
      previousFocus?.focus()
    }
  }, [onClose])

  if (!content || typeof document === "undefined") return null

  return createPortal(
    <div className="info-modal-overlay">
      <div className="info-modal-dialog" role="dialog" aria-modal="true" aria-labelledby="info-modal-title" aria-describedby="info-modal-definition">
        <div className="info-modal-title-bar">
          <h2 id="info-modal-title">{content.title}</h2>
        </div>
        <div className="info-modal-panel">
          <button ref={closeButtonRef} type="button" className="info-modal-close" aria-label={`Close ${content.title} information`} onClick={onClose}>
            <X aria-hidden="true" />
          </button>
          <div className="info-modal-scroll">
            <section className="info-modal-section">
              <h3>Definition</h3>
              <p id="info-modal-definition">{content.definition}</p>
            </section>
            <section className="info-modal-section">
              <h3>Formula</h3>
              <p className="info-modal-formula">{content.formula}</p>
              <p className="info-modal-where-label">Where:</p>
              <ul>
                {content.where.map(([term, description]) => <li key={term}><strong>{term}</strong> = {description}</li>)}
              </ul>
            </section>
            <section className="info-modal-section">
              <h3>Interpretation</h3>
              {content.interpretationIntro && <p>{content.interpretationIntro}</p>}
              {Array.isArray(content.interpretation)
                ? <ul>{content.interpretation.map((item) => <li key={item}>{item}</li>)}</ul>
                : <p>{content.interpretation}</p>}
            </section>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

export default InfoModal
