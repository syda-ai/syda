import { createContext, ReactNode, useCallback, useContext, useMemo, useState } from 'react'
import Modal from './Modal'

type AlertOptions = {
  title: string
  message: ReactNode
  okText?: string
}

type ConfirmOptions = {
  title: string
  message: ReactNode
  confirmText?: string
  cancelText?: string
}

type ModalContextValue = {
  alert: (opts: AlertOptions) => Promise<void>
  confirm: (opts: ConfirmOptions) => Promise<boolean>
}

const ModalContext = createContext<ModalContextValue | undefined>(undefined)

type ActiveModal =
  | { type: 'alert'; opts: AlertOptions; resolve: () => void }
  | { type: 'confirm'; opts: ConfirmOptions; resolve: (result: boolean) => void }
  | null

export function ModalProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<ActiveModal>(null)

  const alert = useCallback((opts: AlertOptions) => {
    return new Promise<void>((resolve) => {
      setActive({ type: 'alert', opts, resolve })
    })
  }, [])

  const confirm = useCallback((opts: ConfirmOptions) => {
    return new Promise<boolean>((resolve) => {
      setActive({ type: 'confirm', opts, resolve })
    })
  }, [])

  const value = useMemo<ModalContextValue>(() => ({ alert, confirm }), [alert, confirm])

  const handleClose = () => {
    if (!active) return
    if (active.type === 'alert') {
      active.resolve()
    } else {
      active.resolve(false)
    }
    setActive(null)
  }

  const handleConfirm = () => {
    if (!active || active.type !== 'confirm') return
    active.resolve(true)
    setActive(null)
  }

  return (
    <ModalContext.Provider value={value}>
      {children}
      {active?.type === 'alert' && (
        <Modal
          isOpen
          variant="alert"
          title={active.opts.title}
          message={active.opts.message}
          okText={active.opts.okText}
          onClose={handleClose}
        />
      )}
      {active?.type === 'confirm' && (
        <Modal
          isOpen
          variant="confirm"
          title={active.opts.title}
          message={active.opts.message}
          confirmText={active.opts.confirmText}
          cancelText={active.opts.cancelText}
          onConfirm={handleConfirm}
          onClose={handleClose}
        />
      )}
    </ModalContext.Provider>
  )
}

export function useModal() {
  const ctx = useContext(ModalContext)
  if (!ctx) {
    throw new Error('useModal must be used within a ModalProvider')
  }
  return ctx
}


