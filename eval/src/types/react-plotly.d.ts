declare module 'react-plotly.js' {
  import { Component } from 'react'
  import { Data, Layout } from 'plotly.js'

  interface PlotProps {
    data: Data[]
    layout?: Partial<Layout>
    config?: any
    style?: React.CSSProperties
    className?: string
    onInitialized?: (figure: any) => void
    onUpdate?: (figure: any) => void
    onPurge?: (figure: any) => void
    onError?: (err: any) => void
    onRelayout?: (eventData: any) => void
    onRedraw?: () => void
    onSelected?: (eventData: any) => void
    onSelecting?: (eventData: any) => void
    onUnselect?: () => void
    onHover?: (eventData: any) => void
    onUnhover?: (eventData: any) => void
    onClick?: (eventData: any) => void
    onDoubleClick?: (eventData: any) => void
    onDoubleTap?: (eventData: any) => void
    onBeforeHover?: (eventData: any) => void
    onHoverLabel?: (eventData: any) => void
    onRelayouting?: (eventData: any) => void
    onRestyle?: (eventData: any) => void
    onDeselect?: () => void
    onSunburstClick?: (eventData: any) => void
    onSunburstHover?: (eventData: any) => void
    onSunburstUnhover?: (eventData: any) => void
    onSunburstSelect?: (eventData: any) => void
    onSunburstDeselect?: () => void
    onSunburstRestyle?: (eventData: any) => void
    onSunburstRelayout?: (eventData: any) => void
    onSunburstDoubleClick?: (eventData: any) => void
    onSunburstDoubleTap?: (eventData: any) => void
    onSunburstBeforeHover?: (eventData: any) => void
    onSunburstHoverLabel?: (eventData: any) => void
    onSunburstRelayouting?: (eventData: any) => void
    onSunburstRestyle?: (eventData: any) => void
    onSunburstDeselect?: () => void
  }

  export default class Plot extends Component<PlotProps> {}
} 