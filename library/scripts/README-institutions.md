# Institution Extraction System

This script extracts institution information from research papers, similar to how author extraction works.

## Available Scripts

### `extract-institutions-from-papers.ts`

Extracts institution data from papers that have arXiv metadata and populates the `institutions` field.

## Usage

```bash
# Extract institutions from papers
DATABASE_URL="postgres://morganm:password@localhost/metta_library" npm run extract-institutions
```

## How It Works

### 1. **Institution Detection**

The script analyzes arXiv paper metadata and uses pattern matching to identify:

- **Universities**: "Stanford University", "MIT", "UC Berkeley"
- **Research Institutes**: "Google Research", "DeepMind", "OpenAI"
- **Corporate Labs**: "Microsoft Research", "Facebook AI Research"
- **Government Labs**: "NIST", "NASA Ames"

### 2. **Institution Normalization**

- Removes department prefixes ("Department of Computer Science" → "Stanford University")
- Standardizes naming conventions
- Deduplicates similar variations

### 3. **Data Population**

- Updates the `institutions: String[]` field on Paper records
- Links papers to their affiliated institutions
- Provides institution statistics and top institutions list

## Current Limitations

### **Institution Detection Challenges**

Since arXiv doesn't have structured institution data, we use heuristic pattern matching:

- **Limited Accuracy**: May miss institutions with non-standard naming
- **False Positives**: May capture department names as institutions
- **Missing Context**: Can't distinguish primary vs secondary affiliations

### **Potential Improvements**

1. **Enhanced Pattern Recognition**:

```typescript
// More sophisticated institution patterns
const patterns = {
  universities: /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+University\b/g,
  institutes:
    /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+Institute(?:\s+of\s+Technology)?\b/g,
  corporate:
    /\b(Google|Microsoft|Meta|Apple|Amazon|IBM|Intel)\s*(?:Research|AI|Labs?)?\b/g,
};
```

2. **External Data Sources**:

- **ROR (Research Organization Registry)**: Authoritative institution database
- **CrossRef API**: Publishers often include institution metadata
- **ORCID**: Author-institution affiliations
- **Semantic Scholar**: Has institution extraction capabilities

3. **Machine Learning Approach**:

```typescript
// Example: Use ML model for institution extraction
const extractInstitutionsML = async (paperText: string) => {
  // Named Entity Recognition for institutions
  const entities = await nlpModel.extractEntities(paperText, "ORG");
  return entities.filter(isAcademicInstitution);
};
```

## Future Enhancements

### **Institution Model**

Consider creating a proper Institution entity:

```prisma
model Institution {
  id          String   @id @default(cuid())
  name        String   @unique
  aliases     String[] // Alternative names
  country     String?
  city        String?
  type        InstitutionType
  website     String?
  rorId       String?  // Research Organization Registry ID

  papers      PaperInstitution[]
  authors     Author[] // Authors affiliated with this institution

  createdAt   DateTime @default(now())
  updatedAt   DateTime @updatedAt
}

model PaperInstitution {
  paperId       String
  institutionId String
  isPrimary     Boolean @default(false)

  paper         Paper       @relation(fields: [paperId], references: [id])
  institution   Institution @relation(fields: [institutionId], references: [id])

  @@id([paperId, institutionId])
}

enum InstitutionType {
  UNIVERSITY
  RESEARCH_INSTITUTE
  CORPORATE_LAB
  GOVERNMENT_LAB
  HOSPITAL
  NON_PROFIT
}
```

### **Enhanced Institution Detection**

1. **ROR Integration**:

```typescript
const findInstitutionByROR = async (name: string) => {
  const response = await fetch(
    `https://api.ror.org/organizations?query=${name}`
  );
  const data = await response.json();
  return data.items[0]; // Best match
};
```

2. **CrossRef API**:

```typescript
const getInstitutionsFromCrossRef = async (doi: string) => {
  const response = await fetch(`https://api.crossref.org/works/${doi}`);
  const work = await response.json();
  return work.message.author.map((author) => author.affiliation);
};
```

3. **ORCID Integration**:

```typescript
const getAuthorInstitutions = async (orcidId: string) => {
  const response = await fetch(
    `https://api.orcid.org/v3.0/${orcidId}/employments`
  );
  return response.data.affiliations;
};
```

## Integration with Overlay Stack

Once institutions are properly extracted, they'll work seamlessly with your overlay navigation system:

```typescript
// In InstitutionOverlay.tsx
const institutionData = await fetchInstitutionData(institutionName);
// Shows papers, authors, collaboration networks, funding, etc.
```

## Running the Script

### Prerequisites

1. **Database**: PostgreSQL with Prisma schema
2. **Environment**: `.env.local` with `DATABASE_URL`
3. **Papers**: Papers with arXiv links in the database

### Example Run

```bash
$ npm run extract-institutions

🏛️ Extracting institutions from papers...
📊 Found 245 papers without institution data

📦 Processing batch 1/82

📄 Processing: Neural Cellular Automata for Pattern Formation...
   arXiv ID: 2103.08737
  ✅ Updated institutions: University of Copenhagen, DeepMind

📄 Processing: Differentiable Programming for Earth System Modeling...
   arXiv ID: 2005.04240
  ✅ Updated institutions: MIT, Google Research

🎉 Institution extraction completed!
✅ Successfully processed: 240 papers
❌ Errors: 5 papers
🏛️ Total institutions extracted: 342
📊 Papers with institutions: 240

🏆 Top institutions:
   1. Stanford University (23 papers)
   2. MIT (19 papers)
   3. Google Research (16 papers)
   4. UC Berkeley (14 papers)
   5. DeepMind (12 papers)
```

This gives you the foundation for institution-based navigation and analysis in your research library!
