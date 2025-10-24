# Styling Guide

This document outlines the styling patterns and design tokens for the Library project.

## Design Tokens

All design tokens are defined in `tailwind.config.ts` and should be used consistently across the application.

### Colors

Use semantic color names instead of hardcoded colors:

#### Primary (Blue)

- **Use for**: Primary actions, links, focus states
- **Classes**: `bg-primary-500`, `text-primary-600`, `border-primary-300`
- **Examples**: Submit buttons, primary CTAs, active navigation

#### Success (Green)

- **Use for**: Success messages, positive states, confirmations
- **Classes**: `bg-success-500`, `text-success-700`, `border-success-300`
- **Examples**: Success alerts, checkmarks, completed states

#### Danger (Red)

- **Use for**: Errors, destructive actions, warnings
- **Classes**: `bg-danger-500`, `text-danger-700`, `border-danger-300`
- **Examples**: Error alerts, delete buttons, validation errors

#### Warning (Yellow/Orange)

- **Use for**: Warnings, cautions, important notices
- **Classes**: `bg-warning-500`, `text-warning-700`, `border-warning-300`
- **Examples**: Warning alerts, pending states

#### Neutral (Gray)

- **Use for**: Text, borders, backgrounds, disabled states
- **Classes**: `bg-neutral-100`, `text-neutral-600`, `border-neutral-200`
- **Examples**: Body text, card backgrounds, dividers

### Spacing

Use consistent spacing tokens:

```tsx
// Padding & Margin
px-md py-sm  // Standard form field padding
p-lg         // Standard card padding
gap-md       // Standard gap between elements
space-y-lg   // Standard vertical spacing

// Custom spacing
xs: 4px
sm: 8px
md: 16px
lg: 24px
xl: 32px
2xl: 48px
3xl: 64px
```

### Typography

Standard text sizes with built-in line heights:

```tsx
text-xs   // 12px - Labels, captions
text-sm   // 14px - Secondary text, small UI elements
text-base // 16px - Body text (default)
text-lg   // 18px - Subheadings
text-xl   // 20px - Section headers
text-2xl  // 24px - Page titles
text-3xl  // 30px - Hero text
text-4xl  // 36px - Display text
```

### Border Radius

```tsx
rounded - sm; // Subtle rounding (4px)
rounded; // Standard rounding (8px)
rounded - lg; // Card rounding (12px)
rounded - xl; // Modal rounding (16px)
rounded - full; // Pills, avatars
```

### Elevation (Box Shadows)

```tsx
shadow - sm; // Subtle lift
shadow; // Standard card elevation
shadow - md; // Hover states
shadow - lg; // Modals, popovers
shadow - xl; // High priority overlays
```

## Component Patterns

### Buttons

```tsx
// Primary button
<Button className="bg-primary-600 hover:bg-primary-700 text-white">
  Primary Action
</Button>

// Destructive button
<Button className="bg-danger-600 hover:bg-danger-700 text-white">
  Delete
</Button>

// Secondary button
<Button variant="outline">
  Cancel
</Button>
```

### Alerts & Messages

Use the standardized components:

```tsx
import { ErrorAlert } from "@/components/ui/error-alert";
import { LoadingState } from "@/components/ui/loading-spinner";
import { EmptyState } from "@/components/ui/empty-state";

// Error
<ErrorAlert
  title="Error"
  message="Something went wrong"
  onRetry={handleRetry}
/>

// Loading
<LoadingState message="Loading data..." />

// Empty
<EmptyState
  title="No results"
  description="Try adjusting your filters"
  action={{ label: "Clear filters", onClick: handleClear }}
/>
```

### Cards

```tsx
<div className="p-lg rounded-lg border border-neutral-200 bg-white shadow-md">
  {/* Card content */}
</div>
```

### Forms

```tsx
// Form field spacing
<div className="space-y-lg">
  <FormField ... />
  <FormField ... />
</div>

// Input styling (handled by components)
<Input />  // Uses consistent padding, border, focus states

// Error messages
<FormMessage />  // Red text with danger color
```

## Migration Guidelines

When updating existing components:

1. **Replace hardcoded colors** with semantic tokens:

   - `bg-red-50` → `bg-danger-50`
   - `text-blue-600` → `text-primary-600`
   - `border-green-300` → `border-success-300`

2. **Use standard spacing** instead of arbitrary values:

   - `p-6` → `p-lg`
   - `gap-4` → `gap-md`
   - `mt-8` → `mt-xl`

3. **Replace inline alert/error UI** with standard components:

   - Custom error divs → `<ErrorAlert />`
   - Loading spinners → `<LoadingState />`
   - Empty states → `<EmptyState />`

4. **Use consistent shadows**:
   - Card elevation → `shadow-md`
   - Hover states → `shadow-lg`
   - Modals → `shadow-xl`

## Examples

### Before

```tsx
<div className="rounded-lg border border-red-200 bg-red-50 p-3">
  <div className="flex">
    <svg className="h-5 w-5 text-red-400">...</svg>
    <p className="text-sm text-red-700">{error}</p>
  </div>
</div>
```

### After

```tsx
<ErrorAlert message={error} />
```

### Before

```tsx
<button className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700">
  Submit
</button>
```

### After

```tsx
<Button className="bg-primary-600 hover:bg-primary-700">Submit</Button>
```

## Best Practices

1. **Always use design tokens** from the Tailwind config
2. **Prefer semantic component props** over custom styling
3. **Keep spacing consistent** using the defined scale
4. **Use the elevation system** for depth and hierarchy
5. **Maintain color consistency** with semantic naming
6. **Test accessibility** with sufficient color contrast
7. **Document** any new patterns you introduce

## Resources

- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [Design Tokens Specification](./tailwind.config.ts)
- [Component Library](./src/components/ui/)
