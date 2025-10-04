import * as React from "react";
import { cn } from "@/lib/utils";

interface TabsContextValue {
  value: string;
  setValue: (value: string) => void;
}

const TabsContext = React.createContext<TabsContextValue | null>(null);

interface TabsProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
}

export function Tabs({ value, defaultValue, onValueChange, className, children, ...props }: TabsProps) {
  const [internal, setInternal] = React.useState(defaultValue ?? "");
  const isControlled = value !== undefined;
  const currentValue = isControlled ? value : internal;

  const setValue = React.useCallback(
    (next: string) => {
      if (!isControlled) {
        setInternal(next);
      }
      onValueChange?.(next);
    },
    [isControlled, onValueChange]
  );

  const context = React.useMemo<TabsContextValue>(() => ({ value: currentValue, setValue }), [currentValue, setValue]);

  React.useEffect(() => {
    if (!isControlled && defaultValue !== undefined) {
      setInternal(defaultValue);
    }
  }, [defaultValue, isControlled]);

  return (
    <TabsContext.Provider value={context}>
      <div className={cn("space-y-4", className)} {...props}>
        {children}
      </div>
    </TabsContext.Provider>
  );
}

export const TabsList = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("inline-flex h-10 items-center justify-center rounded-md bg-slate-100 p-1", className)} {...props} />
  )
);
TabsList.displayName = "TabsList";

interface TabsTriggerProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  value: string;
}

export const TabsTrigger = React.forwardRef<HTMLButtonElement, TabsTriggerProps>(({ className, value, ...props }, ref) => {
  const context = React.useContext(TabsContext);
  if (!context) throw new Error("TabsTrigger must be used within Tabs");
  const isActive = context.value === value;

  return (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        isActive ? "bg-white shadow" : "text-slate-500",
        className
      )}
      type="button"
      onClick={(event) => {
        props.onClick?.(event);
        if (!event.defaultPrevented) {
          context.setValue(value);
        }
      }}
      aria-selected={isActive}
      {...props}
    />
  );
});
TabsTrigger.displayName = "TabsTrigger";

interface TabsContentProps extends React.HTMLAttributes<HTMLDivElement> {
  value: string;
}

export const TabsContent = React.forwardRef<HTMLDivElement, TabsContentProps>(({ className, value, ...props }, ref) => {
  const context = React.useContext(TabsContext);
  if (!context) throw new Error("TabsContent must be used within Tabs");
  const hidden = context.value !== value;

  return (
    <div
      ref={ref}
      role="tabpanel"
      hidden={hidden}
      className={cn("focus:outline-none", className)}
      {...props}
    />
  );
});
TabsContent.displayName = "TabsContent";
