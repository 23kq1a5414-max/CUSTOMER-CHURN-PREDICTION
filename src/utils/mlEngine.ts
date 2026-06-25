import { ModelType, Hyperparameters, TrainingMetrics, CustomerInput } from '../types';

/**
 * Normalizes scores or returns realistic simulated outcomes based on hyperparameter configurations.
 * This simulates how a real model responds to over/under-fitting.
 */
export function simulateModelTraining(
  modelType: ModelType,
  params: Hyperparameters,
  datasetId: string
): TrainingMetrics {
  const lr = params.learningRate || 0.05;
  const depth = params.maxDepth || 6;
  const estimators = params.numEstimators || 100;
  const reg = params.regularization || 'l2';

  // Base metrics depending on dataset and model
  let baseAuc = 0.81;
  let baseAcc = 0.80;
  let baseRec = 0.62;
  let basePre = 0.68;

  if (datasetId === 'fintech') {
    baseAuc = 0.84; baseAcc = 0.83; baseRec = 0.58; basePre = 0.72;
  } else if (datasetId === 'saas') {
    baseAuc = 0.89; baseAcc = 0.88; baseRec = 0.69; basePre = 0.75;
  }

  // Adjustments based on Model Type
  if (modelType === 'random_forest') {
    baseAuc += 0.03; baseAcc += 0.02; baseRec += 0.05;
  } else if (modelType === 'gradient_boosting') {
    baseAuc += 0.05; baseAcc += 0.04; baseRec += 0.06; basePre += 0.02;
  } else if (modelType === 'decision_tree') {
    baseAuc -= 0.04; baseAcc -= 0.03; baseRec -= 0.05; basePre -= 0.03;
  }

  // Hyperparameter adjustments
  // Learning rate effects
  if (lr > 0.3) {
    // Learning rate too high - unstable training, worse metrics
    baseAuc -= 0.06; baseAcc -= 0.05; baseRec -= 0.08;
  } else if (lr < 0.005) {
    // Learning rate too low - underfitting
    baseAuc -= 0.08; baseAcc -= 0.07; baseRec -= 0.10;
  }

  // Max Depth effects (Overfitting simulation)
  if (depth > 12) {
    if (modelType === 'decision_tree' || modelType === 'gradient_boosting') {
      // Overfitting reduces test metrics
      baseAuc -= 0.05; baseAcc -= 0.04; baseRec -= 0.03;
    } else {
      // Forests are more robust to overfitting
      baseAuc += 0.01; baseAcc += 0.005;
    }
  } else if (depth < 3) {
    // Underfitting
    baseAuc -= 0.07; baseAcc -= 0.06; baseRec -= 0.10;
  }

  // Regularization effects
  if (reg === 'none' && (modelType === 'logistic_regression' || lr > 0.1)) {
    baseAuc -= 0.02; baseAcc -= 0.01;
  }

  // Clamp metrics
  const clamp = (val: number) => Math.max(0.4, Math.min(0.98, val));
  const rAuc = clamp(baseAuc);
  const rAcc = clamp(baseAcc);
  const rRec = clamp(baseRec);
  const rPre = clamp(basePre);
  const rF1 = (2 * rPre * rRec) / (rPre + rRec);

  // Generate Feature Importance list based on dataset
  let featureImportance: Array<{ feature: string; importance: number }> = [];
  if (datasetId === 'telecom') {
    featureImportance = [
      { feature: 'Contract Length (Month-to-Month)', importance: 0.35 * (depth / 6) },
      { feature: 'Tenure (Months)', importance: 0.28 * (estimators / 100) },
      { feature: 'Fiber Optic Internet Service', importance: 0.18 },
      { feature: 'Monthly Charges', importance: 0.11 },
      { feature: 'Tech Support Availability', importance: 0.08 }
    ];
  } else if (datasetId === 'fintech') {
    featureImportance = [
      { feature: 'Customer Age', importance: 0.42 * (depth / 6) },
      { feature: 'Active Member Status', importance: 0.25 },
      { feature: 'Number of Products', importance: 0.18 },
      { feature: 'Account Balance', importance: 0.10 },
      { feature: 'Credit Score', importance: 0.05 }
    ];
  } else {
    featureImportance = [
      { feature: 'Customer Support Tickets', importance: 0.44 * (depth / 6) },
      { feature: 'Monthly Logins / Seat', importance: 0.24 },
      { feature: 'License Utilization Rate', importance: 0.16 },
      { feature: 'Usage Growth Trend', importance: 0.11 },
      { feature: 'Contract Length', importance: 0.05 }
    ];
  }

  // Sort feature importances
  featureImportance.sort((a, b) => b.importance - a.importance);
  // Normalize to sum to 1.0
  const sum = featureImportance.reduce((acc, f) => acc + f.importance, 0);
  featureImportance = featureImportance.map(f => ({
    feature: f.feature,
    importance: Math.round((f.importance / sum) * 100) / 100
  }));

  // Confusion matrix generation based on test size of 1000 records
  const total = 1000;
  const churnRatio = datasetId === 'telecom' ? 0.26 : datasetId === 'fintech' ? 0.20 : 0.15;
  const actualChurn = Math.round(total * churnRatio);
  const actualRetained = total - actualChurn;

  // True Positives (churned correctly predicted)
  const truePositive = Math.round(actualChurn * rRec);
  const falseNegative = actualChurn - truePositive;

  // False Positives (retained predicted as churned)
  // Precision = TP / (TP + FP) => TP + FP = TP / Precision => FP = (TP / Precision) - TP
  const predictedPositive = Math.min(actualRetained, Math.round(truePositive / rPre));
  const falsePositive = Math.max(0, predictedPositive - truePositive);
  const trueNegative = actualRetained - falsePositive;

  // Generate Loss Curve
  const lossCurve = [];
  let currentTrainLoss = 0.8;
  let currentValLoss = 0.82;
  const epochs = modelType === 'logistic_regression' ? 25 : 35;

  for (let i = 1; i <= epochs; i++) {
    // Loss decrease rates depend on hyperparams
    const decayRate = lr * (modelType === 'gradient_boosting' ? 1.2 : 0.9);
    currentTrainLoss -= currentTrainLoss * decayRate * (1 - i / (epochs * 1.5));
    
    // Add noise
    currentTrainLoss += (Math.random() - 0.5) * 0.01;
    if (currentTrainLoss < 0.15) currentTrainLoss = 0.15;

    // Overfitting simulation
    if (depth > 12 && i > epochs * 0.6) {
      currentValLoss += (Math.random() * 0.015); // validation loss diverges
    } else {
      currentValLoss = currentTrainLoss + 0.02 + (Math.random() - 0.5) * 0.01;
    }
    
    lossCurve.push({
      epoch: i,
      loss: Math.round(currentTrainLoss * 1000) / 1000,
      valLoss: Math.round(currentValLoss * 1000) / 1000
    });
  }

  // Generate ROC Curve points
  const rocCurve = [{ fpr: 0, tpr: 0 }];
  const steps = 15;
  // rocAuc dictates the bulge of the curve
  for (let i = 1; i < steps; i++) {
    const fpr = i / steps;
    // TPR mathematically bounded
    const tpr = Math.min(1.0, Math.pow(fpr, 1 - rAuc) + (Math.random() - 0.5) * 0.015);
    rocCurve.push({
      fpr: Math.round(fpr * 100) / 100,
      tpr: Math.max(0, Math.min(1.0, Math.round(tpr * 100) / 100))
    });
  }
  rocCurve.push({ fpr: 1, tpr: 1 });

  return {
    accuracy: Math.round(rAcc * 100) / 100,
    precision: Math.round(rPre * 100) / 100,
    recall: Math.round(rRec * 100) / 100,
    f1Score: Math.round(rF1 * 100) / 100,
    rocAuc: Math.round(rAuc * 100) / 100,
    confusionMatrix: {
      trueNegative,
      falsePositive,
      falseNegative,
      truePositive
    },
    featureImportance,
    lossCurve,
    rocCurve
  };
}

/**
 * Computes individual customer churn risk mathematically using a Logistic Regression formula
 * calibrated to Telecom data trends.
 */
export function estimateTelecomChurnRisk(input: CustomerInput): {
  probability: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  factors: string[];
} {
  // Base log-odds (intercept)
  let logOdds = -1.2;

  // Add contract weights
  if (input.contract === 'month-to-month') {
    logOdds += 1.4;
  } else if (input.contract === 'two-year') {
    logOdds -= 1.1;
  } else {
    logOdds -= 0.3; // one-year
  }

  // Tenure weight (longer tenure = lower churn)
  logOdds -= (input.tenure / 12) * 0.45;

  // Monthly charges weight
  if (input.monthlyCharges > 85) {
    logOdds += 0.5;
  } else if (input.monthlyCharges < 35) {
    logOdds -= 0.4;
  }

  // Internet service weights
  if (input.internetService === 'fiber_optic') {
    logOdds += 0.8; // Fiber optic users churn faster due to premium billing
  } else if (input.internetService === 'no') {
    logOdds -= 0.6;
  }

  // Tech support availability
  if (input.techSupport === 'yes') {
    logOdds -= 0.55;
  } else if (input.techSupport === 'no') {
    logOdds += 0.4;
  }

  // Paperless billing
  if (input.paperlessBilling === 'yes') {
    logOdds += 0.25;
  }

  // Payment style
  if (input.paymentMethod === 'electronic_check') {
    logOdds += 0.5;
  } else if (input.paymentMethod === 'credit_card') {
    logOdds -= 0.3;
  }

  // Demographic nuances
  if (input.seniorCitizen === 'yes') {
    logOdds += 0.2;
  }
  if (input.partner === 'yes' && input.dependents === 'yes') {
    logOdds -= 0.25;
  }

  // Sigmoid transform: Probability = 1 / (1 + e^-z)
  const probability = 1 / (1 + Math.exp(-logOdds));

  // Risk Classification
  let riskLevel: 'Low' | 'Medium' | 'High' = 'Low';
  if (probability > 0.65) riskLevel = 'High';
  else if (probability > 0.35) riskLevel = 'Medium';

  // Identify risk drivers or mitigators
  const factors: string[] = [];
  if (input.contract === 'month-to-month') {
    factors.push('Contract style is flexible Month-to-Month (Major Risk Factor).');
  } else if (input.contract === 'two-year') {
    factors.push('Committed to stable 2-Year Contract (Strong Retention Safeguard).');
  }

  if (input.tenure < 6) {
    factors.push(`Very short tenure of ${input.tenure} months increases early attrition risk.`);
  } else if (input.tenure > 36) {
    factors.push(`Long tenure of ${input.tenure} months demonstrates high brand loyalty.`);
  }

  if (input.techSupport === 'no' && input.internetService !== 'no') {
    factors.push('Lacks Tech Support services, leading to unguided service friction.');
  } else if (input.techSupport === 'yes') {
    factors.push('Subscribed to Tech Support, creating helpful support touchpoints.');
  }

  if (input.internetService === 'fiber_optic') {
    factors.push('Premium Fiber Optic service billing with high monthly charge exposure.');
  }

  if (input.paymentMethod === 'electronic_check') {
    factors.push('Uses manual Electronic Check payments, prone to billing issues.');
  }

  return {
    probability: Math.round(probability * 100),
    riskLevel,
    factors
  };
}

/**
 * True Client-Side Machine Learning Parser & Trainer!
 * This class parses uploaded CSV strings and trains a Logistic Regression model via Stochastic Gradient Descent.
 */
export class ClientSideMLEngine {
  static parseCSV(csvText: string): { headers: string[]; rows: any[] } {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
    if (lines.length === 0) return { headers: [], rows: [] };

    // Parse headers
    const headers = lines[0].split(',').map(h => h.replace(/^["']|["']$/g, '').trim());
    const rows: any[] = [];

    for (let i = 1; i < lines.length; i++) {
      // Basic comma splitter handling quotes
      const values: string[] = [];
      let inQuotes = false;
      let currentVal = '';

      for (let charIndex = 0; charIndex < lines[i].length; charIndex++) {
        const char = lines[i][charIndex];
        if (char === '"' || char === "'") {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          values.push(currentVal.trim());
          currentVal = '';
        } else {
          currentVal += char;
        }
      }
      values.push(currentVal.trim());

      // Map to object
      const row: any = {};
      let hasData = false;
      headers.forEach((header, index) => {
        const rawVal = values[index] || '';
        let cleanVal: any = rawVal.replace(/^["']|["']$/g, '').trim();
        // Convert to number if numeric
        if (cleanVal !== '' && !isNaN(Number(cleanVal))) {
          cleanVal = Number(cleanVal);
        }
        row[header] = cleanVal;
        if (rawVal !== '') hasData = true;
      });

      if (hasData) {
        rows.push(row);
      }
    }

    return { headers, rows };
  }

  static findTargetColumn(headers: string[]): string | null {
    const keywords = ['churn', 'exited', 'exit', 'leave', 'left', 'terminated', 'target', 'label', 'class'];
    for (const h of headers) {
      const lower = h.toLowerCase();
      if (keywords.some(kw => lower.includes(kw))) {
        return h;
      }
    }
    return null;
  }

  static profileDataset(headers: string[], rows: any[], targetCol: string) {
    const profile: Record<string, { type: 'numeric' | 'categorical'; missing: number; uniques?: number; min?: number; max?: number }> = {};
    
    headers.forEach(h => {
      let numericCount = 0;
      let stringCount = 0;
      let missingCount = 0;
      const uniques = new Set<any>();
      let min = Infinity;
      let max = -Infinity;

      rows.forEach(r => {
        const val = r[h];
        if (val === undefined || val === null || val === '') {
          missingCount++;
        } else if (typeof val === 'number') {
          numericCount++;
          uniques.add(val);
          if (val < min) min = val;
          if (val > max) max = val;
        } else {
          stringCount++;
          uniques.add(val);
        }
      });

      const isNumeric = numericCount >= stringCount;
      profile[h] = {
        type: isNumeric ? 'numeric' : 'categorical',
        missing: missingCount,
        uniques: uniques.size,
        ...(isNumeric && numericCount > 0 ? { min, max } : {})
      };
    });

    return profile;
  }

  /**
   * Trains a Logistic Regression model client-side using standard Gradient Descent!
   */
  static trainLogisticRegression(
    rows: any[],
    targetCol: string,
    headers: string[],
    lr = 0.1,
    epochs = 40
  ): {
    metrics: TrainingMetrics;
    success: boolean;
    error?: string;
  } {
    try {
      if (rows.length < 15) {
        throw new Error('Insufficient records to train model (minimum 15 records required).');
      }

      // 1. Identify columns and separate features
      const features = headers.filter(h => h !== targetCol);
      const numericFeatures: string[] = [];
      const categoricalFeatures: string[] = [];

      const profile = this.profileDataset(headers, rows, targetCol);
      features.forEach(f => {
        if (profile[f].type === 'numeric') {
          numericFeatures.push(f);
        } else {
          categoricalFeatures.push(f);
        }
      });

      // 2. Prepare target vector (binary 0/1 mapping)
      // Check labels
      const targetUniques = new Set<any>();
      rows.forEach(r => {
        if (r[targetCol] !== undefined && r[targetCol] !== '') {
          targetUniques.add(r[targetCol]);
        }
      });

      const targetVals = Array.from(targetUniques);
      if (targetVals.length !== 2) {
        throw new Error(`Target column "${targetCol}" must have exactly 2 distinct classes to train Binary Logistic Regression. Found: [${targetVals.join(', ')}]`);
      }

      // Map lower/negative class to 0, higher/positive class to 1.
      // E.g., Yes/No => No=0, Yes=1. Churned/Retained => Retained=0, Churned=1.
      const positiveClassVal = targetVals.find(v => {
        const s = String(v).toLowerCase();
        return s === 'yes' || s === 'true' || s === '1' || s === 'churn' || s === 'churned' || s === 'exited' || s === 'exit';
      }) ?? targetVals[0];
      
      const negativeClassVal = targetVals.find(v => v !== positiveClassVal);

      // 3. Preprocess and impute missing data, standardized scaling, one-hot encode categoricals
      // Gather unique categories for categoricals
      const categoriesMap: Record<string, any[]> = {};
      categoricalFeatures.forEach(f => {
        const uniques = new Set<any>();
        rows.forEach(r => {
          if (r[f] !== undefined && r[f] !== '') uniques.add(r[f]);
        });
        categoriesMap[f] = Array.from(uniques);
      });

      // Scale constants
      const means: Record<string, number> = {};
      const stds: Record<string, number> = {};
      numericFeatures.forEach(f => {
        let sum = 0;
        let count = 0;
        rows.forEach(r => {
          const v = r[f];
          if (typeof v === 'number') {
            sum += v; count++;
          }
        });
        const mean = count > 0 ? sum / count : 0;
        
        let sumSq = 0;
        rows.forEach(r => {
          const v = r[f];
          if (typeof v === 'number') {
            sumSq += Math.pow(v - mean, 2);
          }
        });
        const std = count > 1 ? Math.sqrt(sumSq / (count - 1)) : 1;
        means[f] = mean;
        stds[f] = std === 0 ? 1 : std;
      });

      // 4. Transform Dataset to numeric matrices
      // Columns in model: numeric features + one-hot elements
      const modelFeatureNames: string[] = [];
      numericFeatures.forEach(f => modelFeatureNames.push(f));
      categoricalFeatures.forEach(f => {
        categoriesMap[f].forEach(val => {
          modelFeatureNames.push(`${f}_${val}`);
        });
      });

      const designMatrix: number[][] = [];
      const targets: number[] = [];

      rows.forEach(r => {
        const x: number[] = [];
        
        // Numerics scaled
        numericFeatures.forEach(f => {
          const val = typeof r[f] === 'number' ? r[f] : means[f]; // Impute mean
          x.push((val - means[f]) / stds[f]);
        });

        // Categoricals encoded
        categoricalFeatures.forEach(f => {
          const val = r[f];
          categoriesMap[f].forEach(catVal => {
            x.push(val === catVal ? 1.0 : 0.0);
          });
        });

        designMatrix.push(x);
        targets.push(r[targetCol] === positiveClassVal ? 1.0 : 0.0);
      });

      // 5. Split train/test (80% / 20%)
      const trainSize = Math.floor(designMatrix.length * 0.8);
      
      // Shuffle indices
      const indices = Array.from({ length: designMatrix.length }, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
      }

      const trainX = indices.slice(0, trainSize).map(i => designMatrix[i]);
      const trainY = indices.slice(0, trainSize).map(i => targets[i]);
      const valX = indices.slice(trainSize).map(i => designMatrix[i]);
      const valY = indices.slice(trainSize).map(i => targets[i]);

      // 6. Train model via SGD!
      const numFeatures = modelFeatureNames.length;
      const weights = new Array<number>(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1);
      let bias = 0.0;

      const lossCurve: Array<{ epoch: number; loss: number; valLoss: number }> = [];

      const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

      for (let epoch = 1; epoch <= epochs; epoch++) {
        let trainLossSum = 0;
        
        // Gradient descent step
        for (let i = 0; i < trainX.length; i++) {
          const x = trainX[i];
          const y = trainY[i];

          // Compute prediction
          let z = bias;
          for (let f = 0; f < numFeatures; f++) {
            z += x[f] * weights[f];
          }
          const p = sigmoid(z);

          // Log loss clamping
          const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));
          trainLossSum += -(y * Math.log(pClamped) + (1 - y) * Math.log(1 - pClamped));

          // Gradients
          const error = p - y;
          bias -= lr * error;
          for (let f = 0; f < numFeatures; f++) {
            weights[f] -= lr * error * x[f];
          }
        }

        const avgTrainLoss = trainLossSum / trainX.length;

        // Compute validation loss
        let valLossSum = 0;
        for (let i = 0; i < valX.length; i++) {
          const x = valX[i];
          const y = valY[i];
          let z = bias;
          for (let f = 0; f < numFeatures; f++) {
            z += x[f] * weights[f];
          }
          const p = sigmoid(z);
          const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));
          valLossSum += -(y * Math.log(pClamped) + (1 - y) * Math.log(1 - pClamped));
        }
        const avgValLoss = valLossSum / valX.length;

        lossCurve.push({
          epoch,
          loss: Math.round(avgTrainLoss * 1000) / 1000,
          valLoss: Math.round(avgValLoss * 1000) / 1000
        });
      }

      // 7. Evaluate on validation set
      let trueNegative = 0;
      let falsePositive = 0;
      let falseNegative = 0;
      let truePositive = 0;

      const predictions: number[] = [];
      const probs: number[] = [];

      for (let i = 0; i < valX.length; i++) {
        const x = valX[i];
        const y = valY[i];
        let z = bias;
        for (let f = 0; f < numFeatures; f++) {
          z += x[f] * weights[f];
        }
        const prob = sigmoid(z);
        const pred = prob >= 0.5 ? 1.0 : 0.0;

        probs.push(prob);
        predictions.push(pred);

        if (y === 0.0 && pred === 0.0) trueNegative++;
        else if (y === 0.0 && pred === 1.0) falsePositive++;
        else if (y === 1.0 && pred === 0.0) falseNegative++;
        else if (y === 1.0 && pred === 1.0) truePositive++;
      }

      const totalVal = valX.length;
      const accuracy = (truePositive + trueNegative) / totalVal;
      
      const valPositivePredictionsCount = truePositive + falsePositive;
      const precision = valPositivePredictionsCount > 0 ? truePositive / valPositivePredictionsCount : 0.0;
      
      const valActualPositiveCount = truePositive + falseNegative;
      const recall = valActualPositiveCount > 0 ? truePositive / valActualPositiveCount : 0.0;
      
      const f1Score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0.0;

      // Calculate simple ROC-AUC
      // Get FPR and TPR for multiple thresholds
      const rocCurve: Array<{ fpr: number; tpr: number }> = [{ fpr: 0, tpr: 0 }];
      const thresholds = Array.from({ length: 21 }, (_, i) => i / 20); // 0 to 1
      
      thresholds.forEach(thresh => {
        if (thresh === 0 || thresh === 1) return;
        let tp = 0; let fp = 0; let fn = 0; let tn = 0;
        for (let i = 0; i < valX.length; i++) {
          const prob = probs[i];
          const actual = valY[i];
          const pred = prob >= thresh ? 1.0 : 0.0;

          if (actual === 0.0 && pred === 0.0) tn++;
          else if (actual === 0.0 && pred === 1.0) fp++;
          else if (actual === 1.0 && pred === 0.0) fn++;
          else if (actual === 1.0 && pred === 1.0) tp++;
        }
        const tprVal = (tp + fn) > 0 ? tp / (tp + fn) : 0.0;
        const fprVal = (tn + fp) > 0 ? fp / (tn + fp) : 0.0;
        rocCurve.push({ fpr: fprVal, tpr: tprVal });
      });
      rocCurve.push({ fpr: 1, tpr: 1 });
      // Sort ROC curve points by FPR ascending
      rocCurve.sort((a, b) => a.fpr - b.fpr);

      // Trapeze rule for AUC
      let rocAuc = 0.0;
      for (let i = 0; i < rocCurve.length - 1; i++) {
        rocAuc += 0.5 * (rocCurve[i + 1].fpr - rocCurve[i].fpr) * (rocCurve[i + 1].tpr + rocCurve[i].tpr);
      }
      rocAuc = Math.min(1.0, Math.max(0.5, rocAuc));

      // 8. Generate Feature Importances from trained weights
      // Sort feature weights by absolute importance
      const featureImportance = modelFeatureNames.map((name, idx) => ({
        feature: name,
        importance: Math.abs(weights[idx])
      }));
      featureImportance.sort((a, b) => b.importance - a.importance);
      
      // Top 5 features normalized
      const top5 = featureImportance.slice(0, 5);
      const topSum = top5.reduce((sumVal, f) => sumVal + f.importance, 0);
      const normalizedTop5 = top5.map(f => ({
        feature: f.feature,
        importance: topSum > 0 ? Math.round((f.importance / topSum) * 100) / 100 : 0.2
      }));

      return {
        success: true,
        metrics: {
          accuracy: Math.round(accuracy * 100) / 100,
          precision: Math.round(precision * 100) / 100,
          recall: Math.round(recall * 100) / 100,
          f1Score: Math.round(f1Score * 100) / 100,
          rocAuc: Math.round(rocAuc * 100) / 100,
          confusionMatrix: {
            trueNegative,
            falsePositive,
            falseNegative,
            truePositive
          },
          featureImportance: normalizedTop5,
          lossCurve,
          rocCurve
        }
      };
    } catch (e: any) {
      return {
        success: false,
        metrics: {} as any,
        error: e.message || 'An unknown error occurred during SGD training.'
      };
    }
  }
}
