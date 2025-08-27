"use client"

import * as React from "react"
import { Check, X } from "lucide-react"
import { ToastContainer, toast as toastifyToast } from "react-toastify"
import "react-toastify/dist/ReactToastify.css"

export function Toaster() {
  return (
    <ToastContainer
      position="top-right"
      autoClose={5000}
      hideProgressBar={false}
      newestOnTop
      closeOnClick
      rtl={false}
      pauseOnFocusLoss
      draggable
      pauseOnHover
      theme="light"
      className="[--toastify-toast-min-height:64px]"
      toastClassName={() =>
        "relative flex p-1 mb-2 min-h-16 rounded-md justify-between overflow-hidden cursor-pointer bg-background text-foreground shadow-lg border"
      }
      bodyClassName={() => "text-sm flex gap-3 items-center p-3"}
      icon={false}
    />
  )
}

type ToastType = "success" | "error" | "info" | "warning"

export function toast(
  message: string,
  type: ToastType = "info",
  options: {
    title?: string
    description?: string
    action?: React.ReactNode
  } = {}
) {
  const { title, description, action } = options

  const toastId = React.useId()
  
  const getIcon = () => {
    switch (type) {
      case "success":
        return <Check className="h-5 w-5 text-green-500" />
      case "error":
        return <X className="h-5 w-5 text-red-500" />
      default:
        return null
    }
  }

  const content = (
    <div className="flex flex-col">
      {title && <div className="font-semibold">{title}</div>}
      <div className="text-muted-foreground">{description || message}</div>
      {action && <div className="mt-2">{action}</div>}
    </div>
  )

  switch (type) {
    case "success":
      return toastifyToast.success(content, {
        icon: getIcon(),
        toastId,
      })
    case "error":
      return toastifyToast.error(content, {
        icon: getIcon(),
        toastId,
      })
    case "warning":
      return toastifyToast.warning(content, {
        icon: getIcon(),
        toastId,
      })
    default:
      return toastifyToast.info(content, {
        icon: getIcon(),
        toastId,
      })
  }
}
