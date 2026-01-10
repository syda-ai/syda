import { ReactNode, useEffect } from 'react'
import './Modal.css'

type ModalVariant = 'alert' | 'confirm'

type BaseProps = {
  isOpen: boolean
  title: string
  message: ReactNode
  onClose: () => void
}

type ConfirmExtras = {
  variant: 'confirm'
  confirmText?: string
  cancelText?: string
  onConfirm: () => void
}

type AlertExtras = {
  variant: 'alert'
  okText?: string
}

type ModalProps = BaseProps & (ConfirmExtras | AlertExtras)

export default function Modal(props: ModalProps) {
  const { isOpen, title, message, onClose } = props

  useEffect(() => {
    if (!isOpen) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="modal-overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="modal">
        <div className="modal-header">
          <h3 className="modal-title">{title}</h3>
          <button className="modal-close" aria-label="Close" onClick={onClose}>✖</button>
        </div>
        <div className="modal-body">
          {typeof message === 'string' ? <p style={{ margin: 0 }}>{message}</p> : message}
        </div>
        <div className="modal-footer">
          {props.variant === 'confirm' ? (
            <>
              <button className="btn secondary" onClick={onClose}>
                {props.cancelText || 'Cancel'}
              </button>
              <button className="btn" onClick={props.onConfirm}>
                {props.confirmText || 'Confirm'}
              </button>
            </>
          ) : (
            <button className="btn" onClick={onClose}>
              {props.okText || 'OK'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}


