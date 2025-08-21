# LLM-Based Adobe Object Selection

This document explains the new LLM-based alternative approach for selecting Adobe PDF objects that correspond to desired figure images.

## Overview

The existing PDF figure extraction system uses fragile logic to map desired figures (like "Figure 1", "Figure 2b", "Figure 2c") to Adobe PDF objects. This new implementation provides an LLM-based alternative that directly analyzes the raw Adobe data to make these selections.

## How It Works

1. **OpenAI Vision Analysis**: First, OpenAI's vision model analyzes the PDF and identifies key figures like `["Figure 1", "Figure 2b", "Figure 2c"]`

2. **LLM Object Selection**: Instead of using complex heuristic logic, an LLM (Claude) analyzes the raw Adobe data structure and selects the appropriate objects for each desired figure

3. **Existing Conversion**: The selected objects are then processed using the existing solid conversion logic to extract figure images

## Configuration

To use the LLM approach instead of the traditional semantic mapping:

```bash
export USE_LLM_ADOBE_SELECTION=true
```

The system will automatically fall back to the traditional approach if the LLM selection fails.

## Implementation Details

### New Function: `selectAdobeObjectsWithLLM`

This function takes:

- `keyFigures`: Array of figures identified by OpenAI vision (e.g., `[{figureNumber: "Figure 1", ...}, {figureNumber: "Figure 2b", ...}]`)
- `rawAdobeElements`: All Adobe PDF elements with properties like ObjectID, Page, Path, Text, Bounds

The LLM analyzes this data and returns a mapping of ObjectIDs to figure identifiers.

### LLM Prompt Strategy

The LLM receives:

- List of desired figure identifiers
- Simplified Adobe element structure (ObjectID, page, path, text snippet, bounds, element type)
- Guidelines for selecting appropriate objects

### Schema

```typescript
const AdobeObjectSelectionSchema = z.object({
  selections: z.array(
    z.object({
      figureIdentifier: z.string(), // e.g., "Figure 1", "Figure 2a"
      selectedObjectID: z.number(), // Adobe ObjectID
      confidence: z.enum(["high", "medium", "low"]),
      reasoning: z.string(), // Explanation of selection
    })
  ),
  globalReasoning: z.string(), // Overall process explanation
});
```

## Testing

To test the new approach:

1. Set the environment variable:

   ```bash
   export USE_LLM_ADOBE_SELECTION=true
   ```

2. Process a PDF through the existing extraction pipeline

3. Look for console logs showing LLM reasoning:
   ```
   🤖 Using LLM-based Adobe object selection...
   ✅ LLM selected 3 Adobe objects
   🧠 LLM reasoning: [explanation]
   📍 Figure 1 → ObjectID 1234 (high) - [specific reasoning]
   ```

## Benefits

- **Intelligent Selection**: LLM can understand spatial relationships, text content, and figure structure
- **Adaptive**: Works with various PDF layouts and figure arrangements
- **Transparent**: Provides reasoning for each selection
- **Robust**: Falls back to traditional approach if needed
- **Non-Disruptive**: Keeps existing conversion logic intact

## Future Improvements

- Add caching for LLM selections to improve performance
- Fine-tune prompt based on real-world testing results
- Add confidence thresholds for automatic fallback
- Support for additional figure types (tables, equations, etc.)
