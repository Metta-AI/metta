// Copy-pasted from gridworks
'use client'
import { flip, Placement, useFloating, useHover, useInteractions } from '@floating-ui/react'
import { FC, ReactNode, useState } from 'react'

export const Tooltip: FC<{
  children: ReactNode
  placement?: Placement
  render: () => ReactNode
}> = ({ children, placement = 'top-start', render }) => {
  const [isOpen, setIsOpen] = useState(false)

  const { refs, floatingStyles, context } = useFloating({
    open: isOpen,
    onOpenChange: setIsOpen,
    placement,
    middleware: [flip()],
  })

  const hover = useHover(context)

  const { getReferenceProps, getFloatingProps } = useInteractions([hover])
  return (
    <>
      <div ref={refs.setReference} {...getReferenceProps()}>
        {children}
      </div>
      {isOpen && (
        <div
          ref={refs.setFloating}
          {...getFloatingProps({
            style: floatingStyles,
            className: 'z-50 rounded-lg border border-zinc-400 bg-white px-3 py-1.5 text-sm shadow-xl',
          })}
        >
          {render()}
        </div>
      )}
    </>
  )
}
